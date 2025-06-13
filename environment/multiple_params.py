import os
import gymnasium as gym
import numpy as np
import random
import types # For MethodType
import json # For saving results
import time
import itertools

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 

from substrate_network import SubstrateNetwork 
from vnr_generator import generate_vnr
from vne_env import VNEEnv 

# --- Configuration ---
MODEL_PATH = "five_model/models/ppo_vne_plain_train_lr0.0003_nsteps2048_nepochs10_gamma0.99_batch64_ent0.001_5Msteps/final_model.zip" 
TOTAL_SIMULATION_VNRS_TO_PROCESS = 1000 # VNRs to process for EACH parameter combination
RE_EMBED_BROKEN_VNRS = True 

# --- Parameter Sweep Grids ---
#VNR_INTER_ARRIVAL_MEAN_LIST = [100/2, 100/3, 100/4, 100/5, 100/6] 
VNR_INTER_ARRIVAL_MEAN_LIST = [100/4]
#VNR_MEAN_TTL_LIST = [250]      
#VNR_MEAN_TTL_LIST = [50, 100, 150, 200, 250, 300, 350, 400] # Mean TTL values for VNRs
VNR_MEAN_TTL_LIST = [100, 200, 300, 400, 500, 600, 700, 800, 900] # Mean TTL values for VNRs   

# --- Seeding ---
SIMULATION_SEED = 42
SUBSTRATE_SEED_BASE = 123 
VNR_GEN_SEED_START = 0 

RESULTS_DIR = "Final_ttl/" 
os.makedirs(RESULTS_DIR, exist_ok=True) 

# --- Helper function to sanitize data for JSON ---
def sanitize_for_json(data):
    if isinstance(data, dict): return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list): return [sanitize_for_json(i) for i in data]
    elif isinstance(data, set): return [sanitize_for_json(i) for i in sorted(list(data))] 
    elif isinstance(data, (np.float64, np.float32, np.float16, float)): 
        if np.isnan(data) or np.isinf(data): return None  
        return float(data) 
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8, int)): return int(data) 
    elif isinstance(data, np.bool_): return bool(data)
    return data

# --- VNESimulationWrapper (Modified to collect inter-arrival times) ---
class VNESimulationWrapper(gym.Wrapper):
    def __init__(self, env, vnr_inter_arrival_mean): 
        super().__init__(env)
        self.vnr_inter_arrival_mean = vnr_inter_arrival_mean 
        self.current_simulation_time = 0; self.next_vnr_arrival_sim_time = 0 
        self.vne_env = env.unwrapped 
        
        self.inter_arrival_time_data = [] # New: To store generated inter-arrival times
        self.concurrency_data = []; self.snn_removal_data = [] 
        self.cumulative_removed_snn = {'ground': 0, 'air': 0, 'leo': 0}
        self.snn_appearance_data = []; self.cumulative_added_snn = {'ground': 0, 'air': 0, 'leo': 0} 
        self.simulation_events = { "total_vnrs_broken_by_dynamics": 0, "total_vnrs_node_re_embedded_successfully": 0, "re_embedding_attempts": 0}
        self.broken_vnr_progression_data = [] 
        self._schedule_next_vnr_arrival() 

    def _schedule_next_vnr_arrival(self):
        inter_arrival = int(np.random.exponential(scale=self.vnr_inter_arrival_mean));
        if inter_arrival <= 0: inter_arrival = 1 
        self.inter_arrival_time_data.append(inter_arrival) # Store the generated value
        self.next_vnr_arrival_sim_time = self.current_simulation_time + inter_arrival

    def reset(self, **kwargs): 
        self._advance_simulation_time_and_events(); self._schedule_next_vnr_arrival() 
        obs, info = self.env.reset(**kwargs); info['current_simulation_time'] = self.current_simulation_time
        return obs, info

    def step(self, action): 
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated and info.get("vnr_status") == "successfully_embedded":
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
            self.concurrency_data.append({'time': self.current_simulation_time, 'active_vnrs': len(self.vne_env.sn.embedding_mapping)})
            expired_vnr_ids = [vnr_id for vnr_id, expiry_time in list(self.vne_env.sn.expiry_queue.items()) if self.current_simulation_time >= expiry_time]
            for vnr_id_expired_commit in expired_vnr_ids: self.vne_env.sn.release_vnr_resources(vnr_id_expired_commit)
            if self.current_simulation_time > 0: 
                affected_vnr_details_from_sn = self.vne_env.sn.step_dynamics() 
                removed_counts = self.vne_env.sn.get_last_removed_counts(); added_counts = self.vne_env.sn.get_last_added_counts() 
                for domain in ['ground', 'air', 'leo']:
                    self.cumulative_removed_snn[domain] += removed_counts.get(domain, 0)
                    self.cumulative_added_snn[domain] += added_counts.get(domain, 0)
                self.snn_removal_data.append({'time': self.current_simulation_time, **self.cumulative_removed_snn})
                self.snn_appearance_data.append({'time': self.current_simulation_time, **self.cumulative_added_snn})
                if affected_vnr_details_from_sn: 
                    num_affected = len(affected_vnr_details_from_sn)
                    self.simulation_events["re_embedding_attempts"] += num_affected
                    self.simulation_events["total_vnrs_broken_by_dynamics"] += num_affected 
                    if RE_EMBED_BROKEN_VNRS:
                        re_embedded_ids = self.vne_env.sn.reembed_lost_vnns(affected_vnr_details_from_sn)
                        self.simulation_events["total_vnrs_node_re_embedded_successfully"] += len(re_embedded_ids)
                        for vnr_id_broken in affected_vnr_details_from_sn:
                            if vnr_id_broken not in re_embedded_ids: self.vne_env.sn.release_vnr_resources(vnr_id_broken)
                    else:
                        for vnr_id_broken in affected_vnr_details_from_sn: self.vne_env.sn.release_vnr_resources(vnr_id_broken)
                    if num_affected > 0: self.broken_vnr_progression_data.append({'time': self.current_simulation_time, 'cumulative_broken_vnrs': self.simulation_events["total_vnrs_broken_by_dynamics"]})
                if not self.broken_vnr_progression_data or self.broken_vnr_progression_data[-1]['time'] < self.current_simulation_time :
                    last_known_broken = self.broken_vnr_progression_data[-1]['cumulative_broken_vnrs'] if self.broken_vnr_progression_data else 0
                    self.broken_vnr_progression_data.append({'time': self.current_simulation_time, 'cumulative_broken_vnrs': last_known_broken})

# --- Main Simulation Loop ---
def run_parameter_sweep():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}"); return
    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        print(f"PPO model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading PPO model: {e}"); return
        
    param_combinations = list(itertools.product(VNR_INTER_ARRIVAL_MEAN_LIST, VNR_MEAN_TTL_LIST))
    total_runs = len(param_combinations)
    print(f"\nStarting parameter sweep for {total_runs} combinations...")

    for i, (arrival_mean, ttl_mean) in enumerate(param_combinations):
        run_start_time = time.time()
        SIM_RUN_NAME = f"vnrs_{TOTAL_SIMULATION_VNRS_TO_PROCESS}_arr_{arrival_mean}_ttl_{ttl_mean}_reembed_{RE_EMBED_BROKEN_VNRS}_seed_{SIMULATION_SEED}"
        RESULTS_FILE_PATH = os.path.join(RESULTS_DIR, f"sim_results_{SIM_RUN_NAME}.json")

        print(f"\n--- Starting Run {i+1}/{total_runs}: Arrival Mean={arrival_mean}, TTL Mean={ttl_mean} ---")
        
        random.seed(SIMULATION_SEED); np.random.seed(SIMULATION_SEED) 
        sn = SubstrateNetwork(seed=SUBSTRATE_SEED_BASE) 
        
        max_sn_nodes_for_obs = sn.ground_nodes + sn.air_nodes + sn.leo_nodes + 20
        vnr_generation_counter = VNR_GEN_SEED_START
        def vnr_gen_wrapper_sim():
            nonlocal vnr_generation_counter
            vnr = generate_vnr(seed_counter=vnr_generation_counter, ttl=ttl_mean)
            vnr_generation_counter += 1
            return vnr

        base_env = VNEEnv(substrate_network_instance=sn, vnr_generator_func=vnr_gen_wrapper_sim, max_sn_nodes_obs=max_sn_nodes_for_obs) 
        
        def get_last_processed_vnr_details_method(self_env_ref): return getattr(self_env_ref, "_last_processed_vnr_for_wrapper", None)
        VNEEnv.get_last_processed_vnr_details = get_last_processed_vnr_details_method
        original_finalize_method = base_env._finalize_vnr_embedding 
        def modified_finalize_embedding_sim(self_env_ref, successful, reason=""):
            if self_env_ref.current_vnr: self_env_ref._last_processed_vnr_for_wrapper = self_env_ref.current_vnr.copy()
            else: self_env_ref._last_processed_vnr_for_wrapper = None
            original_finalize_method(successful=successful, reason=reason)
        base_env._finalize_vnr_embedding = types.MethodType(modified_finalize_embedding_sim, base_env)

        sim_env_wrapper = VNESimulationWrapper(base_env, arrival_mean)

        model.set_env(sim_env_wrapper)
        vec_env_used_by_model = model.get_env()
        if vec_env_used_by_model is not None: vec_env_used_by_model.seed(SIMULATION_SEED) 
        
        simulation_metrics = {
            "configuration": { "model_path": MODEL_PATH, "total_vnrs_to_process": TOTAL_SIMULATION_VNRS_TO_PROCESS, "vnr_inter_arrival_mean": arrival_mean, "vnr_mean_ttl": ttl_mean, "re_embed_broken_vnrs": RE_EMBED_BROKEN_VNRS, "simulation_seed": SIMULATION_SEED, "substrate_seed": SUBSTRATE_SEED_BASE },
            "total_vnrs_processed_by_agent": 0, "successful_embeddings": 0, "failed_initial_embeddings": 0, "total_revenue": 0.0, "total_cost": 0.0, "vnr_details": [], "plot_data_points": [], 
            "initial_lifetimes": {}, "concurrency_data": [], "snn_removal_data": [], "snn_appearance_data": [], "simulation_events": {}, "broken_vnr_progression_data": [], "inter_arrival_times": [] # New
        }

        obs = vec_env_used_by_model.reset() 
        
        for vnr_idx in range(TOTAL_SIMULATION_VNRS_TO_PROCESS):
            current_vnr_actual_id = sim_env_wrapper.vne_env.current_vnr['id'] if sim_env_wrapper.vne_env.current_vnr else f"vnr_loop_{vnr_idx}"
            current_sim_time_at_vnr_arrival = sim_env_wrapper.current_simulation_time
            current_vnr_details_for_log = sim_env_wrapper.vne_env.current_vnr.copy() if sim_env_wrapper.vne_env.current_vnr else {}
            done_from_vecenv = [False] 
            while not done_from_vecenv[0]: 
                action, _ = model.predict(obs, deterministic=True) 
                new_obs, _, dones_vec, infos_vec = vec_env_used_by_model.step(action)
                obs = new_obs; done_from_vecenv = dones_vec 
                if done_from_vecenv[0]: 
                    info_dict = infos_vec[0]
                    simulation_metrics["total_vnrs_processed_by_agent"] += 1
                    vnr_revenue = info_dict.get('revenue', 0.0); vnr_cost = info_dict.get('cost', 0.0)
                    vnr_result_info = { "vnr_id": current_vnr_actual_id, "status": info_dict.get('vnr_status', 'N/A'), "revenue": vnr_revenue, "cost": vnr_cost, "sim_time_arrival": current_sim_time_at_vnr_arrival, "num_nodes": len(current_vnr_details_for_log.get('nodes', [])), "ttl_actual_sampled": current_vnr_details_for_log.get('ttl', 0) }
                    if info_dict.get('vnr_status') == "successfully_embedded": simulation_metrics["successful_embeddings"] += 1
                    else: simulation_metrics["failed_initial_embeddings"] +=1 
                    simulation_metrics["total_revenue"] += vnr_revenue
                    simulation_metrics["total_cost"] += vnr_cost
                    simulation_metrics["vnr_details"].append(vnr_result_info)
                    break
            if simulation_metrics["total_vnrs_processed_by_agent"] >= TOTAL_SIMULATION_VNRS_TO_PROCESS: break

        simulation_metrics["initial_lifetimes"] = sn.get_all_initial_lifetimes() 
        simulation_metrics["concurrency_data"] = sim_env_wrapper.concurrency_data
        simulation_metrics["snn_removal_data"] = sim_env_wrapper.snn_removal_data
        simulation_metrics["snn_appearance_data"] = sim_env_wrapper.snn_appearance_data 
        simulation_metrics["simulation_events"] = sim_env_wrapper.simulation_events
        simulation_metrics["broken_vnr_progression_data"] = sim_env_wrapper.broken_vnr_progression_data
        simulation_metrics["inter_arrival_times"] = sim_env_wrapper.inter_arrival_time_data # Store collected data
        
        total_processed = simulation_metrics["total_vnrs_processed_by_agent"]
        if total_processed > 0:
            final_acceptance_rate = simulation_metrics["successful_embeddings"] / total_processed
            avg_revenue = simulation_metrics["total_revenue"] / total_processed
            avg_cost = simulation_metrics["total_cost"] / total_processed
            revenue_cost_ratio = simulation_metrics["total_revenue"] / simulation_metrics["total_cost"] if simulation_metrics["total_cost"] > 0 else 0.0
        else: final_acceptance_rate, avg_revenue, avg_cost, revenue_cost_ratio = 0,0,0,0
        simulation_metrics["final_summary"] = { "acceptance_rate": final_acceptance_rate, "average_revenue_per_vnr": avg_revenue, "average_cost_per_vnr": avg_cost, "overall_revenue_to_cost_ratio": revenue_cost_ratio, **simulation_metrics["simulation_events"] }
        
        run_end_time = time.time(); run_duration = run_end_time - run_start_time
        simulation_metrics["final_summary"]["simulation_runtime_seconds"] = run_duration
        print(f"Run finished in {run_duration:.2f} seconds. Acceptance Rate: {final_acceptance_rate:.2%}")

        sanitized_metrics = sanitize_for_json(simulation_metrics)
        with open(RESULTS_FILE_PATH, 'w') as f: json.dump(sanitized_metrics, f, indent=4)
        print(f"Results for this run saved to: {RESULTS_FILE_PATH}")
        
    print("\n--- Parameter Sweep Finished ---")

if __name__ == '__main__':
    run_parameter_sweep()

