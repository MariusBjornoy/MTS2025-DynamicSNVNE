import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
from collections import deque

# --- Diagnostic Configuration ---
# Set to a specific substrate node ID that you know exists at the start.
# We will watch this node's resources. e.g., "ground_1", "air_61", "leo_91"
# Check your SubstrateNetwork._get_next_node_id() to predict an ID.
NODE_TO_WATCH = "leo_91" 
PRINT_DIAGNOSTICS = True # Set to True to enable debug prints

class VNEEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, substrate_network_instance, vnr_generator_func, 
                 alpha_rank=0.5, beta_rank=0.5, max_vnr_nodes_obs=10, max_sn_nodes_obs=100,
                 base_success_reward=1.0, ratio_scaling_factor=0.5, high_efficiency_ratio_bonus=5.0):
        super(VNEEnv, self).__init__()
        # ... (rest of __init__ is the same as before)
        self.sn = substrate_network_instance
        self.vnr_generator_func = vnr_generator_func
        self.alpha_rank = alpha_rank; self.beta_rank = beta_rank   
        self.current_vnr = None; self._last_processed_vnr_for_wrapper = None 
        self.current_vnr_ordered_vnns_to_place = deque() 
        self.current_vnn_to_embed = None; self.vn_to_sn_mapping_current_vnr = {} 
        self.temp_sn_node_reservations = {}; self.temp_sn_link_reservations = [] 
        self.max_vnr_nodes_obs = max_vnr_nodes_obs; self.max_sn_nodes_obs = max_sn_nodes_obs
        self.action_space = spaces.Discrete(self.max_sn_nodes_obs) 
        obs_dim = 4 + (self.max_sn_nodes_obs * 7) 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32) 
        self.substrate_nodes_list = []; self.node_id_to_idx = {}       
        self.current_step_in_vnr_embedding = 0
        self.max_steps_per_vnr_episode = (self.max_vnr_nodes_obs * 2) + 5 
        self.total_successful_vnrs = 0; self.total_failed_vnrs = 0
        self.total_revenue = 0; self.total_cost = 0    
        self.base_success_reward = base_success_reward
        self.ratio_scaling_factor = ratio_scaling_factor
        self.high_efficiency_ratio_bonus = high_efficiency_ratio_bonus

    def _get_substrate_node_list_and_mapping(self):
        self.substrate_nodes_list = sorted(list(self.sn.network.nodes()))
        self.node_id_to_idx = {node_id: i for i, node_id in enumerate(self.substrate_nodes_list)}

    def _calculate_node_rank(self, sn_node_id):
        if sn_node_id not in self.sn.network: return 0
        sn_node_data = self.sn.network.nodes[sn_node_id]
        available_cpu = sn_node_data.get('cpu', 0)
        total_adj_bw = sum(self.sn.network.edges[sn_node_id, neighbor].get('bandwidth', 0) for neighbor in sorted(list(self.sn.network.neighbors(sn_node_id))) if self.sn.network.has_edge(sn_node_id, neighbor))
        return (self.alpha_rank * available_cpu) + (self.beta_rank * total_adj_bw)

    def _get_observation(self):
        # ... (this method remains the same)
        if self.current_vnn_to_embed is None: return np.zeros(self.observation_space.shape, dtype=np.float32)
        obs_parts = []; vnn = self.current_vnn_to_embed
        norm_vnn_cpu = np.clip(vnn['cpu'] / 50.0, 0, 1) 
        obs_parts.extend([ norm_vnn_cpu, 1.0 if 'ground' in vnn['candidate_domains'] else 0.0, 1.0 if 'air' in vnn['candidate_domains'] else 0.0, 1.0 if 'leo' in vnn['candidate_domains'] else 0.0 ])
        sn_node_features_count = 0
        for sn_id_in_list in self.substrate_nodes_list:
            if sn_node_features_count >= self.max_sn_nodes_obs: break
            if sn_id_in_list not in self.sn.network: obs_parts.extend([0.0] * 7); sn_node_features_count += 1; continue
            sn_data = self.sn.network.nodes[sn_id_in_list]; rank = self._calculate_node_rank(sn_id_in_list) 
            is_used = 1.0 if sn_id_in_list in self.vn_to_sn_mapping_current_vnr.values() else 0.0
            norm_cpu = np.clip(sn_data.get('cpu',0) / 100.0, 0, 1); adj_bw = self.sn.get_total_adjacent_bandwidth(sn_id_in_list); norm_adj_bw = np.clip(adj_bw / 500.0, 0, 1); norm_rank = np.clip(rank / 300.0, 0, 1) 
            obs_parts.extend([ norm_cpu, norm_adj_bw, norm_rank, 1.0 if sn_data.get('type')=='g' else 0.0, 1.0 if sn_data.get('type')=='a' else 0.0, 1.0 if sn_data.get('type')=='l' else 0.0, is_used ]); sn_node_features_count += 1
        while sn_node_features_count < self.max_sn_nodes_obs: obs_parts.extend([0.0] * 7); sn_node_features_count += 1
        final_obs = np.array(obs_parts, dtype=np.float32)
        if final_obs.shape[0] != self.observation_space.shape[0]:
            padded_obs = np.zeros(self.observation_space.shape, dtype=np.float32); len_to_copy = min(final_obs.shape[0], self.observation_space.shape[0])
            padded_obs[:len_to_copy] = final_obs[:len_to_copy]; return np.clip(padded_obs, 0.0, 1.0) 
        return np.clip(final_obs, 0.0, 1.0)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.current_vnr = self.vnr_generator_func()
        self._last_processed_vnr_for_wrapper = self.current_vnr.copy() if self.current_vnr else None
        self.current_vnr_ordered_vnns_to_place.clear()
        if self.current_vnr and 'path_ordered_vnn_ids' in self.current_vnr and 'nodes' in self.current_vnr:
            vnn_objects_map = {vnn['id']: vnn for vnn in self.current_vnr['nodes']}
            for vnn_id_in_path in self.current_vnr['path_ordered_vnn_ids']:
                if vnn_id_in_path in vnn_objects_map: self.current_vnr_ordered_vnns_to_place.append(vnn_objects_map[vnn_id_in_path])
        self.vn_to_sn_mapping_current_vnr = {}; self.temp_sn_node_reservations = {}; self.temp_sn_link_reservations = [] 
        if self.current_vnr_ordered_vnns_to_place: self.current_vnn_to_embed = self.current_vnr_ordered_vnns_to_place.popleft()
        else: self.current_vnn_to_embed = None 
        self._get_substrate_node_list_and_mapping(); self.current_step_in_vnr_embedding = 0
        observation = self._get_observation()
        info = {"vnr_id": self.current_vnr['id'] if self.current_vnr else "N/A", "status": "embedding_started"}
        return observation, info

    def step(self, action):
        self.current_step_in_vnr_embedding += 1
        reward = 0.0; terminated = False; truncated = False  
        info = {"vnr_id": self.current_vnr['id'] if self.current_vnr else "N/A"}

        if self.current_vnr is None: 
            terminated = True; info['vnr_status'] = "error_no_current_vnr"; info['revenue'] = 0; info['cost'] = 0
            return self._get_observation(), -1.0, terminated, truncated, info

        if self.current_vnn_to_embed:
            chosen_sn_id = None
            if 0 <= action < len(self.substrate_nodes_list): chosen_sn_id = self.substrate_nodes_list[action]
            
            # --- START DIAGNOSTIC PRINT ---
            if PRINT_DIAGNOSTICS and chosen_sn_id == NODE_TO_WATCH and NODE_TO_WATCH in self.sn.network.nodes:
                print(f"[DEBUG {self.current_vnr['id']}] Agent chose {NODE_TO_WATCH}. Checking CPU. Available: {self.sn.network.nodes[NODE_TO_WATCH]['cpu']}. Required: {self.current_vnn_to_embed['cpu']}")
            # --- END DIAGNOSTIC PRINT ---
            
            vnn = self.current_vnn_to_embed; valid_placement = False; placement_error = "None"
            if chosen_sn_id is None or chosen_sn_id not in self.sn.network:
                reward -= 0.5; placement_error = "SN_node_non_existent_or_invalid_action_index"
            elif self.sn.network.nodes[chosen_sn_id]['type'] not in vnn['candidate_domains']:
                reward -= 0.5; placement_error = "wrong_domain"
            elif self.sn.network.nodes[chosen_sn_id]['cpu'] < vnn['cpu']: 
                reward -= 0.5; placement_error = "insufficient_cpu"
            elif chosen_sn_id in self.vn_to_sn_mapping_current_vnr.values():
                reward -= 0.5; placement_error = "co_location_this_vnr"
            else: valid_placement = True
            
            if valid_placement:
                self.vn_to_sn_mapping_current_vnr[vnn['id']] = chosen_sn_id
                self.temp_sn_node_reservations[chosen_sn_id] = self.temp_sn_node_reservations.get(chosen_sn_id, 0) + vnn['cpu']
                reward += 0.1
            else: reward -= 0.1
            
            if self.current_vnr_ordered_vnns_to_place: self.current_vnn_to_embed = self.current_vnr_ordered_vnns_to_place.popleft()
            else: self.current_vnn_to_embed = None 
        
        if self.current_vnn_to_embed is None: 
            expected_nodes_count = len(self.current_vnr.get('nodes',[])) if self.current_vnr else 0
            if len(self.vn_to_sn_mapping_current_vnr) == expected_nodes_count and expected_nodes_count > 0 :
                links_successfully_embedded, reason = self._embed_links_for_current_vnr()
                if links_successfully_embedded:
                    self._finalize_vnr_embedding(successful=True); reward = self.base_success_reward; info['vnr_status'] = "successfully_embedded"
                else:
                    self._finalize_vnr_embedding(successful=False); reward = -1.0; info['vnr_status'] = f"failed_link_embedding ({reason})"
            else:
                self._finalize_vnr_embedding(successful=False); reward = -1.0; info['vnr_status'] = "failed_incomplete_vnn_mapping"
            terminated = True 
            
        if self.current_step_in_vnr_embedding >= self.max_steps_per_vnr_episode and not terminated:
            truncated = True; terminated = True 
            if info.get('vnr_status') is None : 
                 self._finalize_vnr_embedding(successful=False, reason="max_steps_reached"); reward = -1.0; info['vnr_status'] = "failed_max_steps_reached"
            
        observation = self._get_observation()
        if terminated: 
            # ... (rest of the terminated block is the same as before)
            current_vnr_object = self._last_processed_vnr_for_wrapper 
            is_successful = info.get('vnr_status') == "successfully_embedded"
            vnr_id_for_cost_calc = current_vnr_object['id'] if current_vnr_object and is_successful else None
            info['revenue'] = self.calculate_revenue_for_vnr(current_vnr_object, is_successful)
            info['cost'] = self.calculate_cost_for_vnr(current_vnr_object, is_successful, vnr_id_if_successful=vnr_id_for_cost_calc)
            if is_successful:
                current_revenue_val = info['revenue']; current_cost_val = info['cost']; revenue_cost_ratio = 0.0
                if current_cost_val > 0: revenue_cost_ratio = current_revenue_val / current_cost_val
                elif current_revenue_val > 0: revenue_cost_ratio = self.high_efficiency_ratio_bonus 
                elif current_revenue_val == 0 and current_cost_val == 0: revenue_cost_ratio = 1.0 
                reward += (self.ratio_scaling_factor * revenue_cost_ratio)
            info['acceptance_rate'] = self.get_acceptance_rate(); info['node_utilization'] = self.sn.get_node_utilization()
            info['link_utilization'] = self.sn.get_link_utilization()
        return observation, reward, terminated, truncated, info

    def _embed_links_for_current_vnr(self):
        # ... (this method remains the same)
        if not self.current_vnr: return False, "no_current_vnr_for_links"
        vnr_links_original = self.current_vnr.get('links', [])
        if not vnr_links_original: return True, "no_links_to_embed"
        vnr_links_sorted = sorted(vnr_links_original, key=lambda l: (l.get('from',''), l.get('to','')))
        current_vnr_link_allocations_temp = []; g_temp_view = self.sn.network.copy()
        for vnl in vnr_links_sorted: 
            vnn_from_id = vnl['from']; vnn_to_id = vnl['to']; bw_req = vnl['bandwidth']
            sn_from_node = self.vn_to_sn_mapping_current_vnr.get(vnn_from_id); sn_to_node = self.vn_to_sn_mapping_current_vnr.get(vnn_to_id)
            if not sn_from_node or not sn_to_node: return False, f"vnn_not_mapped_for_link_{vnl.get('id','N/A')}"
            if sn_from_node == sn_to_node: current_vnr_link_allocations_temp.append({'vnl_id': vnl.get('id', f"{vnn_from_id}-{vnn_to_id}"),'path': [sn_from_node],'bw_reserved': 0}); continue
            path = self._bfs_find_path(g_temp_view, sn_from_node, sn_to_node, bw_req)
            if path:
                current_vnr_link_allocations_temp.append({'vnl_id': vnl.get('id', f"{vnn_from_id}-{vnn_to_id}"),'path': path,'bw_reserved': bw_req})
                if len(path) > 1:
                    for i in range(len(path) - 1): u,v = path[i],path[i+1]; g_temp_view.edges[u,v]['bandwidth'] -= bw_req
            else: return False, f"no_path_for_link_{vnl.get('id','N/A')}_from_{sn_from_node}_to_{sn_to_node}_req_{bw_req}BW"
        self.temp_sn_link_reservations = current_vnr_link_allocations_temp
        return True, "all_links_path_found"

    def _bfs_find_path(self, graph_view, sn_source_id, sn_target_id, bw_required):
        # ... (this method remains the same)
        if sn_source_id not in graph_view or sn_target_id not in graph_view: return None
        q = deque([(sn_source_id, [sn_source_id])]); visited = {sn_source_id}
        while q:
            curr, path = q.popleft()
            if curr == sn_target_id: return path 
            sorted_neighbors = sorted(list(graph_view.neighbors(curr))) 
            for neighbor in sorted_neighbors: 
                if neighbor not in visited and graph_view.has_edge(curr, neighbor): 
                    edge_data = graph_view.edges[curr, neighbor]
                    if edge_data.get('bandwidth', 0) >= bw_required:
                        visited.add(neighbor); new_path = list(path); new_path.append(neighbor); q.append((neighbor, new_path))
        return None

    def _finalize_vnr_embedding(self, successful, reason=""):
        vnr_id_to_finalize = self.current_vnr['id'] if self.current_vnr and 'id' in self.current_vnr else "N/A"
        
        if successful and self.current_vnr: 
            for vnn_id_key, sn_node_id_val in self.vn_to_sn_mapping_current_vnr.items():
                original_vnn_obj = next((v for v in self.current_vnr.get('nodes',[]) if v['id'] == vnn_id_key), None)
                if original_vnn_obj: 
                    cpu_req_val = original_vnn_obj['cpu']
                    self.sn.embed_vnn(vnr_id_to_finalize, vnn_id_key, sn_node_id_val, cpu_req_val) 
                    # --- START DIAGNOSTIC PRINT ---
                    if PRINT_DIAGNOSTICS and sn_node_id_val == NODE_TO_WATCH and NODE_TO_WATCH in self.sn.network.nodes:
                        print(f"DEBUG: VNR {vnr_id_to_finalize} SUCCESSFUL. CPU {cpu_req_val} subtracted from {NODE_TO_WATCH}. New CPU: {self.sn.network.nodes[NODE_TO_WATCH]['cpu']}")
                    # --- END DIAGNOSTIC PRINT ---

            for link_alloc_item in self.temp_sn_link_reservations:
                self.sn.embed_vnl(vnr_id_to_finalize, link_alloc_item['vnl_id'], link_alloc_item['path'], link_alloc_item['bw_reserved'])
            self.total_successful_vnrs += 1
        else: self.total_failed_vnrs += 1
        
        self.vn_to_sn_mapping_current_vnr = {}; self.temp_sn_node_reservations = {}; self.temp_sn_link_reservations = []
        self.current_vnn_to_embed = None
        if hasattr(self, 'current_vnr_ordered_vnns_to_place'): self.current_vnr_ordered_vnns_to_place.clear()

    def calculate_revenue_for_vnr(self, vnr_details, successfully_embedded):
        if not successfully_embedded or not vnr_details: return 0.0
        revenue = sum(node.get('cpu', 0) for node in vnr_details.get('nodes', [])) + sum(link.get('bandwidth', 0) for link in vnr_details.get('links', []))
        self.total_revenue += revenue
        return revenue

    def calculate_cost_for_vnr(self, vnr_details, successfully_embedded, vnr_id_if_successful=None):
        if not successfully_embedded or not vnr_details: return 0.0
        cost = sum(node.get('cpu', 0) for node in vnr_details.get('nodes', []))
        if vnr_id_if_successful and vnr_id_if_successful in self.sn.embedding_mapping:
            for link_alloc in self.sn.embedding_mapping[vnr_id_if_successful].get('links', []):
                path = link_alloc.get('path', []); bw_reserved = link_alloc.get('bw_reserved', 0)
                num_hops = len(path) - 1 if path and len(path) > 1 else 0 
                cost += bw_reserved * num_hops 
        else: cost += sum(link.get('bandwidth', 0) for link in vnr_details.get('links', []))
        self.total_cost += cost 
        return cost

    def get_acceptance_rate(self):
        total_processed_by_env = self.total_successful_vnrs + self.total_failed_vnrs
        if total_processed_by_env == 0: return 0.0
        return self.total_successful_vnrs / total_processed_by_env

    def render(self, mode='human'): pass
    def close(self): pass
