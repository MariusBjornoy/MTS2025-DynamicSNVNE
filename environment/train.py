import os
import gymnasium as gym
import numpy as np
import time
import types # For MethodType
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor 
import random

# Assuming these are in the same directory or properly installed
from substrate_network import SubstrateNetwork # Ensure this is the version with internal RNGs and reembed returns set
from vnr_generator import generate_vnr     
from vneEnv import VNEEnv                 # Ensure this is the version with enhanced rewards & deterministic BFS

# --- ثابت Configuration for Training Environment ---
LAMBDA_ARRIVAL_TRAIN = 0.04
VNR_INTER_ARRIVAL_MEAN_TRAIN = 1 / LAMBDA_ARRIVAL_TRAIN # Should be 25
VNR_MEAN_TTL_TRAIN = 250  # Mean TTL for VNRs during training
RE_EMBED_BROKEN_VNRS_TRAIN = True # Enable re-embedding during training

# --- Fixed PPO Hyperparameters for this Training Run ---
# Using a set of standard PPO hyperparameters
LEARNING_RATE = 3e-4  # (0.0003)
N_STEPS = 2048      
N_EPOCHS = 10          
GAMMA = 0.99     
BATCH_SIZE = 64         
ENT_COEF = 0.001 # Small entropy bonus for exploration

PPO_POLICY = "MlpPolicy"
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# --- Training Run Configuration ---
TOTAL_PPO_TIMESTEPS_PER_RUN = 5_000_000 # As requested

# --- Logging and Saving ---
RUN_NAME = f"ppo_vne_plain_train_lr{LEARNING_RATE}_nsteps{N_STEPS}_nepochs{N_EPOCHS}_gamma{GAMMA}_batch{BATCH_SIZE}_ent{ENT_COEF}_5Msteps"
BASE_LOG_DIR = "five_model/"
CURRENT_RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard_logs", RUN_NAME)
CURRENT_RUN_MODEL_DIR = os.path.join(BASE_LOG_DIR, "models", RUN_NAME)
os.makedirs(CURRENT_RUN_LOG_DIR, exist_ok=True)
os.makedirs(CURRENT_RUN_MODEL_DIR, exist_ok=True)

# --- Custom Callback for Logging Metrics to TensorBoard ---
class VNETrainingMetricsCallback(BaseCallback):
    def __init__(self, check_freq, run_name_for_print="N/A", verbose=0): 
        super(VNETrainingMetricsCallback, self).__init__(verbose)
        self.check_freq = check_freq 
        self.run_name_for_print = run_name_for_print

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.training_env.num_envs == 1: 
                current_env_level = self.training_env.envs[0]
                vne_env_instance = None; sim_wrapper_instance = None
                max_unwraps = 5; unwraps = 0
                temp_env = current_env_level
                while hasattr(temp_env, 'env') and unwraps < max_unwraps:
                    if isinstance(temp_env, VNESimulationWrapper):
                        sim_wrapper_instance = temp_env
                        if hasattr(sim_wrapper_instance.env, 'env'): # Check if Monitor is wrapping VNEEnv
                             vne_env_instance = sim_wrapper_instance.env.env # VNESimWrapper -> Monitor -> VNEEnv
                        else: # Should not happen if Monitor is always used as per setup
                             vne_env_instance = sim_wrapper_instance.env 
                        break
                    temp_env = temp_env.env; unwraps += 1
                
                if vne_env_instance is None : 
                    if isinstance(temp_env, VNESimulationWrapper):
                        sim_wrapper_instance = temp_env
                        if hasattr(sim_wrapper_instance.env, 'env'):
                             vne_env_instance = sim_wrapper_instance.env.env
                        else: vne_env_instance = sim_wrapper_instance.env
                    elif isinstance(temp_env, Monitor) and hasattr(temp_env, 'env') and isinstance(temp_env.env, VNEEnv):
                        vne_env_instance = temp_env.env 
                    elif isinstance(temp_env, VNEEnv): 
                         vne_env_instance = temp_env

                if vne_env_instance and isinstance(vne_env_instance, VNEEnv):
                    acceptance_rate = vne_env_instance.get_acceptance_rate()
                    current_total_revenue = vne_env_instance.total_revenue
                    current_total_cost = vne_env_instance.total_cost
                    revenue_to_cost_ratio = current_total_revenue / current_total_cost if current_total_cost > 0 else 0.0
                    
                    self.logger.record("custom_metrics/acceptance_rate", acceptance_rate)
                    self.logger.record("custom_metrics/revenue_to_cost_ratio", revenue_to_cost_ratio)
                    self.logger.record("custom_metrics/node_utilization", vne_env_instance.sn.get_node_utilization())
                    self.logger.record("custom_metrics/link_utilization", vne_env_instance.sn.get_link_utilization())
                    total_processed_env = vne_env_instance.total_successful_vnrs + vne_env_instance.total_failed_vnrs
                    self.logger.record("custom_metrics/total_vnrs_processed_env", total_processed_env)
                    
                    if sim_wrapper_instance: # sim_wrapper_instance is VNESimulationWrapper
                         self.logger.record("custom_metrics/simulation_time_env_wrapper", sim_wrapper_instance.current_simulation_time)
                    
                    if total_processed_env > 0: 
                        vne_env_instance.total_successful_vnrs = 0
                        vne_env_instance.total_failed_vnrs = 0
                        vne_env_instance.total_revenue = 0
                        vne_env_instance.total_cost = 0
        return True

# --- Simplified VNESimulationWrapper for Training ---
class VNESimulationWrapper(gym.Wrapper): 
    def __init__(self, env, vnr_inter_arrival_mean_param):
        super().__init__(env) # env here is Monitor(VNEEnv_instance)
        self.vnr_inter_arrival_mean = vnr_inter_arrival_mean_param
        self.current_simulation_time = 0
        self.next_vnr_arrival_sim_time = 0
        if isinstance(env, Monitor):
            self.vne_env_instance = env.unwrapped 
        else: 
            self.vne_env_instance = env 
        
        self.simulation_events = { 
            "total_vnrs_broken_by_dynamics": 0,
            "total_vnrs_node_re_embedded_successfully": 0, 
            "re_embedding_attempts": 0,
        }
        self._schedule_next_vnr_arrival()

    def _schedule_next_vnr_arrival(self):
        inter_arrival = int(np.random.exponential(scale=self.vnr_inter_arrival_mean))
        if inter_arrival <= 0: inter_arrival = 1
        self.next_vnr_arrival_sim_time = self.current_simulation_time + inter_arrival

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None and hasattr(self.env, 'seed'): self.env.seed(seed) 
        self._advance_simulation_time_and_events()
        self._schedule_next_vnr_arrival()
        obs, info = self.env.reset(**kwargs) 
        if 'current_simulation_time' not in info: info['current_simulation_time'] = self.current_simulation_time
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) 
        if terminated or truncated: 
            if info.get("vnr_status") == "successfully_embedded":
                last_vnr = self.vne_env_instance.get_last_processed_vnr_details() 
                if last_vnr and last_vnr.get('id') == info.get("vnr_id"):
                    vnr_actual_lifetime_val = last_vnr.get('ttl', 1000)
                    expiry_at = self.current_simulation_time + vnr_actual_lifetime_val
                    self.vne_env_instance.sn.register_vnr_ttl(info["vnr_id"], expiry_at)
        info['current_simulation_time'] = self.current_simulation_time
        return obs, reward, terminated, truncated, info

    def _advance_simulation_time_and_events(self):
        while self.current_simulation_time < self.next_vnr_arrival_sim_time:
            self.current_simulation_time += 1
            expired_vnr_ids = []
            for vnr_id_expired, expiry_details in list(self.vne_env_instance.sn.expiry_queue.items()):
                expiry_time = expiry_details if isinstance(expiry_details, (int, float)) else expiry_details.get('time')
                if expiry_time is not None and self.current_simulation_time >= expiry_time: expired_vnr_ids.append(vnr_id_expired)
            for vnr_id_expired_commit in expired_vnr_ids: self.vne_env_instance.sn.release_vnr_resources(vnr_id_expired_commit)

            if self.current_simulation_time > 0: 
                affected_vnr_details_from_sn = self.vne_env_instance.sn.step_dynamics()
                if affected_vnr_details_from_sn:
                    num_affected = len(affected_vnr_details_from_sn)
                    self.simulation_events["re_embedding_attempts"] += num_affected
                    self.simulation_events["total_vnrs_broken_by_dynamics"] += num_affected 
                    if RE_EMBED_BROKEN_VNRS_TRAIN:
                        re_embedded_ids = self.vne_env_instance.sn.reembed_lost_vnns(affected_vnr_details_from_sn)
                        self.simulation_events["total_vnrs_node_re_embedded_successfully"] += len(re_embedded_ids)
                        for vnr_id_broken in affected_vnr_details_from_sn:
                            if vnr_id_broken not in re_embedded_ids: self.vne_env_instance.sn.release_vnr_resources(vnr_id_broken)
                    else:
                        for vnr_id_broken in affected_vnr_details_from_sn: self.vne_env_instance.sn.release_vnr_resources(vnr_id_broken)

# --- Main Training Function ---
def train_single_agent():
    print(f"Starting plain training run: {RUN_NAME}")
    run_start_time = time.time()

    # --- Initialize Environment ---
    sn_train_seed = 731 
    vnr_gen_train_seed_start = 1 
    
    global_rng_seed_for_run_vnr_aspects = 12345 
    random.seed(global_rng_seed_for_run_vnr_aspects); np.random.seed(global_rng_seed_for_run_vnr_aspects)

    sn_instance = SubstrateNetwork(seed=sn_train_seed) 
    max_sn_nodes_for_obs_train = sn_instance.ground_nodes + sn_instance.air_nodes + sn_instance.leo_nodes + 20
    
    vnr_generation_counter_train = vnr_gen_train_seed_start 
    def vnr_gen_wrapper_train():
        nonlocal vnr_generation_counter_train
        vnr = generate_vnr(seed_counter=vnr_generation_counter_train, ttl=VNR_MEAN_TTL_TRAIN)
        vnr_generation_counter_train += 1
        return vnr

    base_env_train = VNEEnv(substrate_network_instance=sn_instance, vnr_generator_func=vnr_gen_wrapper_train, max_sn_nodes_obs=max_sn_nodes_for_obs_train)
    
    def get_last_processed_vnr_details_method(self_env_ref): return getattr(self_env_ref, "_last_processed_vnr_for_wrapper", None)
    VNEEnv.get_last_processed_vnr_details = get_last_processed_vnr_details_method
    original_finalize_method = base_env_train._finalize_vnr_embedding 
    def modified_finalize_embedding_train(self_env_ref, successful, reason=""):
        if self_env_ref.current_vnr: self_env_ref._last_processed_vnr_for_wrapper = self_env_ref.current_vnr.copy()
        else: self_env_ref._last_processed_vnr_for_wrapper = None
        original_finalize_method(successful=successful, reason=reason)
    base_env_train._finalize_vnr_embedding = types.MethodType(modified_finalize_embedding_train, base_env_train)
    
    monitored_vne_env = Monitor(base_env_train, filename=os.path.join(CURRENT_RUN_LOG_DIR, "monitor.csv"))
    sim_env_train = VNESimulationWrapper(monitored_vne_env, VNR_INTER_ARRIVAL_MEAN_TRAIN)
    
    # --- Initialize PPO Agent ---
    agent_seed = 420 
    model = PPO(PPO_POLICY, sim_env_train, learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS, gamma=GAMMA, gae_lambda=PPO_GAE_LAMBDA, clip_range=PPO_CLIP_RANGE,
                ent_coef=ENT_COEF, vf_coef=PPO_VF_COEF, max_grad_norm=PPO_MAX_GRAD_NORM, 
                verbose=1, 
                tensorboard_log=CURRENT_RUN_LOG_DIR, seed=agent_seed)

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=max(1, TOTAL_PPO_TIMESTEPS_PER_RUN // 20), 
                                             save_path=CURRENT_RUN_MODEL_DIR,
                                             name_prefix="ppo_vne_checkpoint", # Removed save_best_only
                                             verbose=1)
    
    metrics_log_freq = max(N_STEPS, 10000) 
    metrics_callback = VNETrainingMetricsCallback(check_freq=metrics_log_freq, run_name_for_print=RUN_NAME) 
    callbacks_list = [checkpoint_callback, metrics_callback]

    print(f"Training agent with fixed params. Logging custom metrics approx every {metrics_callback.check_freq} PPO steps.")
    try:
        model.learn(total_timesteps=TOTAL_PPO_TIMESTEPS_PER_RUN, callback=callbacks_list, tb_log_name="PPO_VNE_run")
    except Exception as e:
        print(f"ERROR during training for run {RUN_NAME}: {e}"); import traceback; traceback.print_exc()
    
    final_model_path = os.path.join(CURRENT_RUN_MODEL_DIR, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model for {RUN_NAME} saved to {final_model_path}")

    run_end_time = time.time(); run_duration = run_end_time - run_start_time
    print(f"Training run {RUN_NAME} finished in {run_duration:.2f} seconds.")
    sim_env_train.close()
    print("\n--- Plain training finished. ---")

if __name__ == '__main__':
    train_single_agent()
