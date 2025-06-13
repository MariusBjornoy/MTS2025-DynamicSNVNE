import os
import gymnasium as gym
import numpy as np
import random
import types # For MethodType
import json # For saving results
import matplotlib.pyplot as plt 
import time # For timing VNR embedding episodes

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 

from substrate_network import SubstrateNetwork 
from vnr_generator import generate_vnr
from vne_env import VNEEnv 

# --- Configuration ---
MODEL_PATH = "five_model/models/ppo_vne_plain_train_lr0.0003_nsteps2048_nepochs10_gamma0.99_batch64_ent0.001_5Msteps/final_model.zip" 
TOTAL_SIMULATION_VNRS_TO_PROCESS = 1000
VNR_INTER_ARRIVAL_MEAN = 100/4
RE_EMBED_BROKEN_VNRS = True 

SEED = 42

SIMULATION_SEED = SEED
SUBSTRATE_SEED = SEED
VNR_GEN_SEED_START = SEED 

RESULTS_DIR = "ppo_vne_simulation_results/" 
os.makedirs(RESULTS_DIR, exist_ok=True) 
MODEL_FILENAME_NO_EXT = MODEL_PATH.split('/')[-1].replace('.zip','')
SIM_RUN_NAME = f"vnrs_{TOTAL_SIMULATION_VNRS_TO_PROCESS}_arr_{VNR_INTER_ARRIVAL_MEAN}_dyn_every_step_reembed_{RE_EMBED_BROKEN_VNRS}_simseed_{SIMULATION_SEED}_substrateseed_{SUBSTRATE_SEED}_vnrseedstart_{VNR_GEN_SEED_START}_model_{MODEL_FILENAME_NO_EXT}"
RESULTS_FILE_NAME = f"sim_results_{SIM_RUN_NAME}.json" 
RESULTS_FILE_PATH = os.path.join(RESULTS_DIR, RESULTS_FILE_NAME)

# --- Helper function to sanitize data for JSON ---
def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    elif isinstance(data, set): 
        return [sanitize_for_json(i) for i in sorted(list(data))] 
    elif isinstance(data, (np.float64, np.float32, np.float16, float)): 
        if np.isnan(data) or np.isinf(data): return None  
        return float(data) 
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8, int)): 
        return int(data) 
    elif isinstance(data, np.bool_):
        return bool(data)
    return data

# --- VNESimulationWrapper (Modified for more data collection) ---
class VNESimulationWrapper(gym.Wrapper):
    def __init__(self, env, vnr_inter_arrival_mean): 
        super().__init__(env)
        self.vnr_inter_arrival_mean = vnr_inter_arrival_mean 
        self.current_simulation_time = 0
        self.next_vnr_arrival_sim_time = 0 
        self.vnrs_processed_in_sim = 0
        self.successful_vnrs_in_sim = 0
        self.vne_env = env.unwrapped 

        self.concurrency_data = [] 
        self.snn_removal_data = [] 
        self.cumulative_removed_snn = {'ground': 0, 'air': 0, 'leo': 0}
        self.snn_appearance_data = [] 
        self.cumulative_added_snn = {'ground': 0, 'air': 0, 'leo': 0} 
        self.simulation_events = { 
            "total_vnrs_broken_by_dynamics": 0, 
            "total_vnrs_node_re_embedded_successfully": 0, 
            "re_embedding_attempts": 0,
        }
        self.broken_vnr_progression_data = [] 
        self._schedule_next_vnr_arrival() 

    def _schedule_next_vnr_arrival(self):
        inter_arrival = int(np.random.exponential(scale=self.vnr_inter_arrival_mean))
        if inter_arrival <= 0: inter_arrival = 1 
        self.next_vnr_arrival_sim_time = self.current_simulation_time + inter_arrival

    def reset(self, **kwargs): 
        self._advance_simulation_time_and_events() 
        self._schedule_next_vnr_arrival() 
        obs, info = self.env.reset(**kwargs) 
        if 'current_simulation_time' not in info:
            info['current_simulation_time'] = self.current_simulation_time
        return obs, info

    def step(self, action): 
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated: 
            self.vnrs_processed_in_sim += 1
            if info.get("vnr_status") == "successfully_embedded":
                self.successful_vnrs_in_sim += 1
                last_vnr = self.vne_env.get_last_processed_vnr_details() 
                if last_vnr and last_vnr.get('id') == info.get("vnr_id"): 
                    vnr_actual_lifetime_val = last_vnr.get('ttl', 1000) 
                    expiry_at = self.current_simulation_time + vnr_actual_lifetime_val 
                    self.vne_env.sn.register_vnr_ttl(info["vnr_id"], expiry_at)
        info['current_simulation_time'] = self.current_simulation_time
        return obs, reward, terminated, truncated, info

    def _advance_simulation_time_and_events(self):
        while self.current_simulation_time < self.next_vnr_arrival_sim_time:
            self.current_simulation_time += 1 
            
            self.concurrency_data.append({
                'time': self.current_simulation_time,
                'active_vnrs': len(self.vne_env.sn.embedding_mapping)
            })

            expired_vnr_ids = []
            for vnr_id_expired, expiry_details in list(self.vne_env.sn.expiry_queue.items()):
                expiry_time = expiry_details if isinstance(expiry_details, (int, float)) else expiry_details.get('time')
                if expiry_time is not None and self.current_simulation_time >= expiry_time:
                    expired_vnr_ids.append(vnr_id_expired)
            for vnr_id_expired_commit in expired_vnr_ids:
                self.vne_env.sn.release_vnr_resources(vnr_id_expired_commit)

            if self.current_simulation_time > 0: 
                affected_vnr_details_from_sn = self.vne_env.sn.step_dynamics() 
                removed_counts_this_step = self.vne_env.sn.get_last_removed_counts() 
                added_counts_this_step = self.vne_env.sn.get_last_added_counts() 

                self.cumulative_removed_snn['ground'] += removed_counts_this_step.get('ground', 0)
                self.cumulative_removed_snn['air'] += removed_counts_this_step.get('air', 0)
                self.cumulative_removed_snn['leo'] += removed_counts_this_step.get('leo', 0)
                self.snn_removal_data.append({
                    'time': self.current_simulation_time,
                    'ground': self.cumulative_removed_snn['ground'],
                    'air': self.cumulative_removed_snn['air'],
                    'leo': self.cumulative_removed_snn['leo']
                })
                
                self.cumulative_added_snn['ground'] += added_counts_this_step.get('ground', 0) 
                self.cumulative_added_snn['air'] += added_counts_this_step.get('air', 0)
                self.cumulative_added_snn['leo'] += added_counts_this_step.get('leo', 0)
                self.snn_appearance_data.append({ 
                    'time': self.current_simulation_time,
                    'ground': self.cumulative_added_snn['ground'],
                    'air': self.cumulative_added_snn['air'],
                    'leo': self.cumulative_added_snn['leo']
                })

                if affected_vnr_details_from_sn: 
                    num_vnrs_affected_this_step = len(affected_vnr_details_from_sn)
                    self.simulation_events["re_embedding_attempts"] += num_vnrs_affected_this_step
                    self.simulation_events["total_vnrs_broken_by_dynamics"] += num_vnrs_affected_this_step 
                    
                    successfully_re_embedded_vnr_ids_this_step = set()
                    if RE_EMBED_BROKEN_VNRS:
                        successfully_re_embedded_vnr_ids_this_step = self.vne_env.sn.reembed_lost_vnns(affected_vnr_details_from_sn)
                        self.simulation_events["total_vnrs_node_re_embedded_successfully"] += len(successfully_re_embedded_vnr_ids_this_step)
                    
                    if RE_EMBED_BROKEN_VNRS:
                        for vnr_id_broken in affected_vnr_details_from_sn: 
                            if vnr_id_broken not in successfully_re_embedded_vnr_ids_this_step:
                                self.vne_env.sn.release_vnr_resources(vnr_id_broken)
                    else: 
                        for vnr_id_broken in affected_vnr_details_from_sn:
                            self.vne_env.sn.release_vnr_resources(vnr_id_broken)
                    
                    if num_vnrs_affected_this_step > 0 : 
                        self.broken_vnr_progression_data.append({
                            'time': self.current_simulation_time,
                            'cumulative_broken_vnrs': self.simulation_events["total_vnrs_broken_by_dynamics"]
                        })
                
                if not self.broken_vnr_progression_data or self.broken_vnr_progression_data[-1]['time'] < self.current_simulation_time :
                    current_broken_count_for_log = self.simulation_events["total_vnrs_broken_by_dynamics"]
                    if self.broken_vnr_progression_data : 
                         current_broken_count_for_log = self.broken_vnr_progression_data[-1]['cumulative_broken_vnrs']
                         if affected_vnr_details_from_sn: 
                              current_broken_count_for_log = self.simulation_events["total_vnrs_broken_by_dynamics"]
                    self.broken_vnr_progression_data.append({
                        'time': self.current_simulation_time,
                        'cumulative_broken_vnrs': current_broken_count_for_log
                    })

# --- Main Simulation Function ---
def run_ppo_simulation():
    print(f"Starting simulation with PPO agent: {MODEL_PATH}")
    print(f"Processing {TOTAL_SIMULATION_VNRS_TO_PROCESS} VNRs.")
    LAMBDA_ARRIVAL = 1/VNR_INTER_ARRIVAL_MEAN 
    VNR_MEAN_TTL_CONFIG = VNR_INTER_ARRIVAL_MEAN * 5 
    print(f"VNR Arrival Lambda: {LAMBDA_ARRIVAL :.2f} (Mean Inter-Arrival: {VNR_INTER_ARRIVAL_MEAN}), Mean VNR TTL: {VNR_MEAN_TTL_CONFIG}, Substrate Dynamics: Every Step, RE_EMBED_BROKEN_VNRS: {RE_EMBED_BROKEN_VNRS}")
    print(f"Simulation Seed: {SIMULATION_SEED}, Substrate Seed: {SUBSTRATE_SEED}, VNR Gen Start Seed: {VNR_GEN_SEED_START}")

    random.seed(SIMULATION_SEED); np.random.seed(SIMULATION_SEED) 
    sn = SubstrateNetwork(seed=SUBSTRATE_SEED) 
    
    max_sn_nodes_for_obs = sn.ground_nodes + sn.air_nodes + sn.leo_nodes + 20
    current_vnr_gen_seed_counter = VNR_GEN_SEED_START
    
    def vnr_gen_wrapper_sim():
        nonlocal current_vnr_gen_seed_counter
        vnr = generate_vnr(seed_counter=current_vnr_gen_seed_counter, ttl=250)
        current_vnr_gen_seed_counter += 1
        return vnr

    base_env = VNEEnv(substrate_network_instance=sn,
                      vnr_generator_func=vnr_gen_wrapper_sim,
                      alpha_rank=0.6, beta_rank=0.4,
                      max_sn_nodes_obs=max_sn_nodes_for_obs,
                      max_vnr_nodes_obs=10) 
    
    def get_last_processed_vnr_details_method(self_env_ref): 
        return getattr(self_env_ref, "_last_processed_vnr_for_wrapper", None)
    VNEEnv.get_last_processed_vnr_details = get_last_processed_vnr_details_method
    original_finalize_method = base_env._finalize_vnr_embedding 
    def modified_finalize_embedding_sim(self_env_ref, successful, reason=""):
        if self_env_ref.current_vnr:
            self_env_ref._last_processed_vnr_for_wrapper = self_env_ref.current_vnr.copy()
        else: self_env_ref._last_processed_vnr_for_wrapper = None
        original_finalize_method(successful=successful, reason=reason)
    base_env._finalize_vnr_embedding = types.MethodType(modified_finalize_embedding_sim, base_env)

    sim_env_wrapper = VNESimulationWrapper(base_env, VNR_INTER_ARRIVAL_MEAN) 

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}"); return
    try:
        model = PPO.load(MODEL_PATH, device="cpu"); model.set_env(sim_env_wrapper) 
        vec_env_used_by_model = model.get_env()
        if vec_env_used_by_model is not None: vec_env_used_by_model.seed(SIMULATION_SEED) 
        print(f"PPO model loaded successfully from {MODEL_PATH} and environment seeded with SIMULATION_SEED={SIMULATION_SEED}.")
    except Exception as e:
        print(f"Error loading PPO model or setting/seeding environment: {e}"); import traceback; traceback.print_exc(); return

    simulation_metrics = {
        "configuration": {
            "model_path": MODEL_PATH, "total_vnrs_to_process": TOTAL_SIMULATION_VNRS_TO_PROCESS,
            "lambda_arrival": LAMBDA_ARRIVAL, 
            "vnr_mean_ttl": VNR_MEAN_TTL_CONFIG,     
            "dynamics_interval": 1, 
            "simulation_seed": SIMULATION_SEED, "substrate_seed": SUBSTRATE_SEED,
            "vnr_gen_seed_start": VNR_GEN_SEED_START, "re_embed_broken_vnrs": RE_EMBED_BROKEN_VNRS
        },
        "total_vnrs_processed_by_agent": 0, "successful_embeddings": 0, "failed_initial_embeddings": 0, 
        "total_revenue": 0.0, "total_cost": 0.0, "vnr_details": [], "plot_data_points": [], 
        "initial_lifetimes": {}, "concurrency_data": [], "snn_removal_data": [], 
        "snn_appearance_data": [], "simulation_events": {}, "broken_vnr_progression_data": []
    }

    obs = vec_env_used_by_model.reset() 
    PLOT_DATA_INTERVAL_VNRS = max(1, TOTAL_SIMULATION_VNRS_TO_PROCESS // 20) 

    for vnr_idx in range(TOTAL_SIMULATION_VNRS_TO_PROCESS):
        current_vnr_actual_id = sim_env_wrapper.env.unwrapped.current_vnr['id'] if sim_env_wrapper.env.unwrapped.current_vnr else f"vnr_loop_{vnr_idx}"
        current_sim_time_at_vnr_arrival = sim_env_wrapper.current_simulation_time
        current_vnr_details_for_log = sim_env_wrapper.env.unwrapped.current_vnr.copy() if sim_env_wrapper.env.unwrapped.current_vnr else {}

        if (vnr_idx + 1) % (TOTAL_SIMULATION_VNRS_TO_PROCESS // 10 if TOTAL_SIMULATION_VNRS_TO_PROCESS >=10 else 1) == 0 :
            print(f"Simulating VNR {vnr_idx + 1}/{TOTAL_SIMULATION_VNRS_TO_PROCESS} (ID: {current_vnr_actual_id}) at SimTime: {current_sim_time_at_vnr_arrival}")
        
        # --- Start timing the VNR embedding process ---
        vnr_embedding_start_time = time.perf_counter()

        vnr_episode_reward = 0; vnr_episode_steps = 0; done_from_vecenv = [False] 

        while not done_from_vecenv[0]: 
            action, _states = model.predict(obs, deterministic=True) 
            new_obs, rewards_vec, dones_vec, infos_vec = vec_env_used_by_model.step(action)
            obs = new_obs; done_from_vecenv = dones_vec 
            info_dict_for_current_step = infos_vec[0]; reward_for_current_step = rewards_vec[0] 
            vnr_episode_reward += reward_for_current_step; vnr_episode_steps += 1
            
            if done_from_vecenv[0]: 
                # --- End timing and calculate duration in milliseconds ---
                vnr_embedding_end_time = time.perf_counter()
                embedding_duration_ms = (vnr_embedding_end_time - vnr_embedding_start_time) * 1000

                simulation_metrics["total_vnrs_processed_by_agent"] += 1
                vnr_revenue = info_dict_for_current_step.get('revenue', 0.0)
                vnr_cost = info_dict_for_current_step.get('cost', 0.0)
                
                vnr_result_info = {
                    "vnr_id": current_vnr_actual_id, 
                    "status": info_dict_for_current_step.get('vnr_status', 'N/A'),
                    "revenue": vnr_revenue, "cost": vnr_cost,
                    "sim_time_arrival": current_sim_time_at_vnr_arrival, 
                    "sim_time_processed": info_dict_for_current_step.get('current_simulation_time', sim_env_wrapper.current_simulation_time),
                    "num_nodes": len(current_vnr_details_for_log.get('nodes', [])), 
                    "num_links": len(current_vnr_details_for_log.get('links', [])),
                    "ttl_actual_sampled": current_vnr_details_for_log.get('ttl', 0),
                    "embedding_time_ms": embedding_duration_ms # Store the calculated time
                }
                if info_dict_for_current_step.get('vnr_status') == "successfully_embedded":
                    simulation_metrics["successful_embeddings"] += 1
                else:
                    simulation_metrics["failed_initial_embeddings"] +=1 
                
                simulation_metrics["total_revenue"] += vnr_revenue
                simulation_metrics["total_cost"] += vnr_cost
                simulation_metrics["vnr_details"].append(vnr_result_info)
                
                if (simulation_metrics["total_vnrs_processed_by_agent"] % PLOT_DATA_INTERVAL_VNRS == 0) or \
                   (simulation_metrics["total_vnrs_processed_by_agent"] == TOTAL_SIMULATION_VNRS_TO_PROCESS):
                    current_total_processed = simulation_metrics["total_vnrs_processed_by_agent"]
                    current_acc_rate = simulation_metrics["successful_embeddings"] / current_total_processed if current_total_processed > 0 else 0
                    current_rev_total = simulation_metrics["total_revenue"]
                    current_cost_total = simulation_metrics["total_cost"] 
                    current_rc_ratio = current_rev_total / current_cost_total if current_cost_total > 0 else 0.0 
                    node_u = base_env.sn.get_node_utilization()
                    link_u = base_env.sn.get_link_utilization()
                    simulation_metrics["plot_data_points"].append({
                        "sim_time": sim_env_wrapper.current_simulation_time, 
                        "vnrs_processed": current_total_processed,
                        "acceptance_rate": current_acc_rate,
                        "revenue_to_cost_ratio": current_rc_ratio,
                        "node_utilization": node_u,
                        "link_utilization": link_u
                    })
                break 
        
        if simulation_metrics["total_vnrs_processed_by_agent"] >= TOTAL_SIMULATION_VNRS_TO_PROCESS:
            break 
    
    simulation_metrics["initial_lifetimes"] = sn.get_all_initial_lifetimes() 
    simulation_metrics["concurrency_data"] = sim_env_wrapper.concurrency_data
    simulation_metrics["snn_removal_data"] = sim_env_wrapper.snn_removal_data
    simulation_metrics["snn_appearance_data"] = sim_env_wrapper.snn_appearance_data 
    simulation_metrics["simulation_events"] = sim_env_wrapper.simulation_events
    simulation_metrics["broken_vnr_progression_data"] = sim_env_wrapper.broken_vnr_progression_data

    total_processed_final = simulation_metrics["total_vnrs_processed_by_agent"]
    if total_processed_final > 0:
        final_acceptance_rate = simulation_metrics["successful_embeddings"] / total_processed_final
        avg_revenue = simulation_metrics["total_revenue"] / total_processed_final
        avg_cost = simulation_metrics["total_cost"] / total_processed_final
        revenue_cost_ratio = simulation_metrics["total_revenue"] / simulation_metrics["total_cost"] if simulation_metrics["total_cost"] > 0 else 0.0 
    else:
        final_acceptance_rate = 0; avg_revenue = 0; avg_cost = 0; revenue_cost_ratio = 0

    simulation_metrics["final_summary"] = {
        "acceptance_rate": final_acceptance_rate,
        "average_revenue_per_vnr": avg_revenue,
        "average_cost_per_vnr": avg_cost,
        "overall_revenue_to_cost_ratio": revenue_cost_ratio,
        "final_node_utilization": base_env.sn.get_node_utilization(),
        "final_link_utilization": base_env.sn.get_link_utilization(),
        "final_overall_utilization": base_env.sn.get_overall_utilization(),
        "simulation_end_time": sim_env_wrapper.current_simulation_time,
        "total_vnrs_embedded_successfully" : simulation_metrics["successful_embeddings"],
        "total_vnrs_failed_initial_embedding" : simulation_metrics["failed_initial_embeddings"] 
    }
    
    print("\n--- Simulation Finished ---")
    # Print general summary
    for key, value in simulation_metrics["final_summary"].items():
        if isinstance(value, float): print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else: print(f"{key.replace('_', ' ').title()}: {value}")

    # Print specific event counts
    sim_events = simulation_metrics.get("simulation_events", {})
    broken_vnrs_count = sim_events.get("total_vnrs_broken_by_dynamics", 0)
    re_embedded_vnrs_count = sim_events.get("total_vnrs_node_re_embedded_successfully", 0)
    
    print(f"Total VNRs Broken by Dynamics: {broken_vnrs_count}")
    if RE_EMBED_BROKEN_VNRS:
        print(f"Total VNRs Successfully Re-embedded (Nodes): {re_embedded_vnrs_count}")
        re_embedding_acceptance_rate = re_embedded_vnrs_count / broken_vnrs_count if broken_vnrs_count > 0 else 0
        print(f"Re-embedding Acceptance Rate for Broken VNRs: {re_embedding_acceptance_rate:.2%}")
    else:
        print("Re-embedding was disabled for this run.")


    # Sanitize the entire metrics dictionary before dumping to JSON
    sanitized_simulation_metrics = sanitize_for_json(simulation_metrics)

    try:
        with open(RESULTS_FILE_PATH, 'w') as f: 
            json.dump(sanitized_simulation_metrics, f, indent=4) 
        print(f"Simulation results saved to: {RESULTS_FILE_PATH}")
    except Exception as e: 
        print(f"Error saving results to JSON: {e}")

    print(f"\nTo generate plots, run: python plot_simulation_results.py \"{RESULTS_FILE_PATH}\"")
    
    sim_env_wrapper.close() 
    if vec_env_used_by_model is not None: vec_env_used_by_model.close()
    print("Environment closed.")

if __name__ == '__main__':
    run_ppo_simulation()
