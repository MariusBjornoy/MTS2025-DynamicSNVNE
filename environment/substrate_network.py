import networkx as nx
import numpy as np
import random 
import math

class SubstrateNetwork:
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            self.rng_np = np.random.RandomState(seed)
            self.rng_py = random.Random(seed) 
        else:
            self.rng_np = np.random.RandomState()
            self.rng_py = random.Random()

        self.ground_nodes = 60; self.air_nodes = 30; self.leo_nodes = 10
        self.added_counts = {'air': 0, 'leo': 0} 
        self.expiry_queue = {}; self.network = nx.Graph()
        self._node_id_counter = 0 
        self.initial_node_lifetimes = {"air": [], "leo": []} 
        self._last_removed_counts_this_step = {'ground': 0, 'air': 0, 'leo': 0}
        self._last_added_counts_this_step = {'ground': 0, 'air': 0, 'leo': 0}
        self._init_nodes(); self._rebuild_edges(); self._patch_isolated_nodes() 
        self.embedding_mapping = {}; self._last_added_nodes = [] 

    def _get_next_node_id(self, prefix):
        self._node_id_counter += 1
        return f"{prefix}_{self._node_id_counter}"

    def _init_nodes(self):
        for _ in range(self.ground_nodes):
            nid = self._get_next_node_id("ground")
            pos = (self.rng_np.uniform(0, 100), self.rng_np.uniform(0, 100), 0)
            cpu = self.rng_np.randint(50, 101)
            self.network.add_node(nid, type="ground", cpu=cpu, initial_cpu=cpu, position=pos)
        for _ in range(self.air_nodes):
            nid = self._get_next_node_id("air")
            pos = (self.rng_np.uniform(0, 100), self.rng_np.uniform(0, 100), self.rng_np.uniform(100, 500))
            cpu = self.rng_np.randint(50, 101); lt = self.rng_np.exponential(scale=5400)
            self.initial_node_lifetimes["air"].append(lt) 
            self.network.add_node(nid, type="air", cpu=cpu, initial_cpu=cpu, position=pos, lifetime=lt, initial_lifetime=lt)
        for _ in range(self.leo_nodes):
            nid = self._get_next_node_id("leo")
            pos = (self.rng_np.uniform(0, 100), self.rng_np.uniform(0, 100), self.rng_np.uniform(500, 2000))
            cpu = self.rng_np.randint(50, 81); lt = self.rng_np.exponential(scale=600)
            self.initial_node_lifetimes["leo"].append(lt) 
            self.network.add_node(nid, type="leo", cpu=cpu, initial_cpu=cpu, position=pos, lifetime=lt, initial_lifetime=lt)
            
    def _incremental_edge_update(self, alpha=0.4, beta=0.2):
        G = self.network; L = 0.0
        if not self._last_added_nodes: return
        pos3d = nx.get_node_attributes(G, 'position'); pos2d = {n: (p[0], p[1]) for n, p in pos3d.items()}
        all_current_nodes_sorted = sorted(list(G.nodes()))
        if len(all_current_nodes_sorted) > 1:
            for i, u_id in enumerate(all_current_nodes_sorted):
                for v_id in all_current_nodes_sorted[i+1:]:
                    if u_id in pos2d and v_id in pos2d: d = math.dist(pos2d[u_id], pos2d[v_id]); L = max(L, d)
        if L == 0: L = 1.0
        existing_nodes_sorted = sorted([n for n in G.nodes() if n not in self._last_added_nodes])
        for u_new in self._last_added_nodes:
            if u_new not in pos2d: continue
            for v_exist in existing_nodes_sorted:
                if v_exist == u_new or G.has_edge(u_new, v_exist) or v_exist not in pos2d: continue
                type_u = G.nodes[u_new]['type']
                type_v = G.nodes[v_exist]['type']
                if (type_u == 'ground' and type_v == 'leo') or (type_u == 'leo' and type_v == 'ground'): continue
                d = math.dist(pos2d[u_new], pos2d[v_exist]); p = beta*math.exp(-d/(alpha*L)) if L>0 else 0.0
                if self.rng_py.random()<p: 
                    bw = self.rng_np.randint(50,81) if (type_u=='air' and type_v=='leo') or (type_u=='leo' and type_v=='air') else self.rng_np.randint(50,101)
                    G.add_edge(u_new,v_exist,bandwidth=bw,initial_bandwidth=bw)
        if len(self._last_added_nodes)>1:
            s_added = sorted(self._last_added_nodes)
            for i,u_new in enumerate(s_added):
                if u_new not in pos2d: continue
                for v_new in s_added[i+1:]:
                    if v_new not in pos2d or G.has_edge(u_new,v_new): continue
                    type_u = G.nodes[u_new]['type']
                    type_v = G.nodes[v_new]['type']
                    if (type_u == 'ground' and type_v == 'leo') or (type_u == 'leo' and type_v == 'ground'): continue
                    d = math.dist(pos2d[u_new], pos2d[v_new]); p = beta*math.exp(-d/(alpha*L)) if L>0 else 0.0
                    if self.rng_py.random()<p: 
                        bw=self.rng_np.randint(50,81) if (type_u=='air' and type_v=='leo') or (type_u=='leo' and type_v=='air') else self.rng_np.randint(50,101)
                        G.add_edge(u_new,v_new,bandwidth=bw,initial_bandwidth=bw)

    def _rebuild_edges(self, alpha=0.4, beta=0.2):
        G = self.network; G.remove_edges_from(list(G.edges())); L = 0.0
        pos3d = nx.get_node_attributes(G, 'position'); pos2d = {n: (p[0], p[1]) for n, p in pos3d.items()}
        nodes_s = sorted(list(G.nodes())); 
        if not nodes_s: return
        if len(nodes_s)>1:
            for i,u_id in enumerate(nodes_s):
                for v_id in nodes_s[i+1:]:
                    if u_id in pos2d and v_id in pos2d: d=math.dist(pos2d[u_id],pos2d[v_id]); L=max(L,d)
        if L==0: L=1.0
        for i,u in enumerate(nodes_s):
            if u not in pos2d: continue
            for v in nodes_s[i+1:]:
                if v not in pos2d: continue
                type_u = G.nodes[u]['type']; type_v = G.nodes[v]['type']
                if (type_u == 'ground' and type_v == 'leo') or (type_u == 'leo' and type_v == 'ground'): continue
                d=math.dist(pos2d[u],pos2d[v]); p=beta*math.exp(-d/(alpha*L)) if L>0 else 0.0
                if self.rng_py.random()<p: 
                    bw=self.rng_np.randint(50,81) if (type_u=='air' and type_v=='leo') or (type_u=='leo' and type_v=='air') else self.rng_np.randint(50,101)
                    G.add_edge(u,v,bandwidth=bw,initial_bandwidth=bw)
        if G.number_of_nodes()>0 and not nx.is_connected(G):
            comps = sorted(list(nx.connected_components(G)), key=lambda c: sorted(list(c))[0])
            main_c = set(comps[0])
            for comp_s_set in comps[1:]: # Changed variable name from comp_s to comp_s_set for clarity
                main_s=sorted(list(main_c)); other_s=sorted(list(comp_s_set)); conn=False
                for u_n in main_s:
                    for v_n in other_s:
                        type_u=G.nodes[u_n]['type']; type_v=G.nodes[v_n]['type']
                        if not ((type_u == 'ground' and type_v == 'leo') or (type_u == 'leo' and type_v == 'ground')):
                            bw=self.rng_np.randint(50,81) if (type_u=='air' and type_v=='leo') or (type_u=='leo' and type_v=='air') else self.rng_np.randint(50,101)
                            G.add_edge(u_n,v_n,bandwidth=bw,initial_bandwidth=bw); conn=True; break
                    if conn: break
                if not conn and main_s and other_s:
                    uf,vf=self.rng_py.choice(main_s),self.rng_py.choice(other_s)
                    bw=self.rng_np.randint(50,101); G.add_edge(uf,vf,bandwidth=bw,initial_bandwidth=bw)
                main_c.update(comp_s_set) # Use comp_s_set here

    def _patch_isolated_nodes(self):
        G=self.network; dom_ord={'ground':0,'air':1,'leo':2}; nodes_s=sorted(list(G.nodes()))
        for n_id in nodes_s:
            if G.degree(n_id)==0:
                node_type,node_domain_idx=G.nodes[n_id]['type'],dom_ord[G.nodes[n_id]['type']]; targets=[]
                for t_id in nodes_s:
                    if n_id==t_id: continue
                    target_type,target_domain_idx=G.nodes[t_id]['type'],dom_ord[G.nodes[t_id]['type']]
                    if abs(node_domain_idx-target_domain_idx)<=1 and not ((node_type=='ground' and target_type=='leo') or (node_type=='leo' and target_type=='ground')): targets.append(t_id)
                if not targets: targets=[_id for _id in nodes_s if _id!=n_id]
                if targets:
                    t_conn=self.rng_py.choice(targets); target_type_chosen=G.nodes[t_conn]['type']
                    bw=self.rng_np.randint(50,81) if (node_type=='air' and target_type_chosen=='leo') or (node_type=='leo' and target_type_chosen=='air') else self.rng_np.randint(50,101)
                    G.add_edge(n_id,t_conn,bandwidth=bw,initial_bandwidth=bw)

    def embed_vnn(self, vnr_id, vnn_id, sn_id, cpu_req):
        if sn_id not in self.network.nodes or self.network.nodes[sn_id]['cpu']<cpu_req: return False
        self.network.nodes[sn_id]['cpu']-=cpu_req
        self.embedding_mapping.setdefault(vnr_id,{'nodes':[],'links':[]})['nodes'].append({'substrate_node':sn_id,'cpu_reserved':cpu_req,'vnn_id':vnn_id})
        return True

    def embed_vnl(self, vnr_id, vnl_id, path, bw_req):
        if not path or len(path)<2: return False
        for i in range(len(path)-1):
            u,v=path[i],path[i+1]
            if not self.network.has_edge(u,v) or self.network.edges[u,v]['bandwidth']<bw_req: return False
        for i in range(len(path)-1): u,v=path[i],path[i+1]; self.network.edges[u,v]['bandwidth']-=bw_req
        self.embedding_mapping.setdefault(vnr_id,{'nodes':[],'links':[]})['links'].append({'path':path,'bw_reserved':bw_req,'vnl_id':vnl_id})
        return True

    def release_vnr_resources(self, vnr_id):
        if vnr_id not in self.embedding_mapping: return
        allocs=self.embedding_mapping[vnr_id]
        for n_alloc in allocs.get('nodes',[]):
            sn_n,cpu_a=n_alloc['substrate_node'],n_alloc['cpu_reserved']
            if sn_n in self.network.nodes: self.network.nodes[sn_n]['cpu']+=cpu_a
        for l_alloc in allocs.get('links',[]):
            p,bw_a=l_alloc['path'],l_alloc['bw_reserved']
            for i in range(len(p)-1):
                u,v=p[i],p[i+1]
                if self.network.has_edge(u,v): self.network.edges[u,v]['bandwidth']+=bw_a
        del self.embedding_mapping[vnr_id]
        if vnr_id in self.expiry_queue: del self.expiry_queue[vnr_id]

    def register_vnr_ttl(self, vnr_id, expiry_ts): self.expiry_queue[vnr_id]=expiry_ts

    def reembed_lost_vnns(self, lost_embeddings_details):
        successfully_re_embedded_vnr_ids = set() 
        sorted_affected_vnr_ids = sorted(list(lost_embeddings_details.keys()))
        for vnr_id in sorted_affected_vnr_ids:
            vnr_data = lost_embeddings_details[vnr_id]; vnns_to_re_embed = vnr_data.get('nodes', [])
            if not vnns_to_re_embed: continue
            num_lost_for_this_vnr = len(vnns_to_re_embed); re_embedded_count_for_this_vnr = 0
            sorted_vnns_to_re_embed = sorted(vnns_to_re_embed, key=lambda x: x.get('vnn_id', ''))
            for lost_vnn_alloc in sorted_vnns_to_re_embed:
                original_vnn_id = lost_vnn_alloc['vnn_id']
                vnn_details = lost_vnn_alloc.get('vnn_details', {})
                cpu_req = vnn_details.get('cpu', lost_vnn_alloc['cpu_reserved']) 
                candidate_types = vnn_details.get('candidate_domains', ['ground', 'air', 'leo'])
                used_sn_nodes_for_this_vnr = set()
                if vnr_id in self.embedding_mapping:
                    for node_alloc in self.embedding_mapping[vnr_id].get('nodes', []):
                        if node_alloc['vnn_id'] != original_vnn_id: used_sn_nodes_for_this_vnr.add(node_alloc['substrate_node'])
                found_new_host_for_this_vnn = False
                for sn_id in sorted(list(self.network.nodes())): 
                    data = self.network.nodes[sn_id]
                    if data['type'] in candidate_types and data['cpu'] >= cpu_req and sn_id not in used_sn_nodes_for_this_vnr: 
                        if self.embed_vnn(vnr_id, original_vnn_id, sn_id, cpu_req):
                            re_embedded_count_for_this_vnr += 1; found_new_host_for_this_vnn = True; break 
                if not found_new_host_for_this_vnn: break 
            if re_embedded_count_for_this_vnr == num_lost_for_this_vnr: successfully_re_embedded_vnr_ids.add(vnr_id) 
        return successfully_re_embedded_vnr_ids 

    def step_dynamics(self):
        self._last_added_nodes = []; affected_vnr_details = {}
        self._last_removed_counts_this_step = {'ground':0,'air':0,'leo':0}
        self._last_added_counts_this_step = {'ground':0,'air':0,'leo':0}
        to_rem=[]; nodes_s=sorted(list(self.network.nodes()))
        for n in nodes_s:
            d=self.network.nodes[n]
            # Use full type names for consistency with how they are stored
            if d['type'] in ('air','leo'): 
                d['lifetime']-=1
                if d['lifetime']<=0: to_rem.append(n)
        
        sorted_to_rem = sorted(to_rem)
        for sn_rem in sorted_to_rem:
            if sn_rem in self.network:
                n_type_rem=self.network.nodes[sn_rem]['type'] # Use full type name
                if n_type_rem in self._last_removed_counts_this_step: self._last_removed_counts_this_step[n_type_rem]+=1
            for vnr_id,allocs in list(self.embedding_mapping.items()):
                lost_nodes_this_vnr_on_sn_rem=[]
                for n_alloc in allocs.get('nodes',[]):
                    if n_alloc['substrate_node']==sn_rem:
                        vnn_detail_for_re_embed = n_alloc.copy()
                        if 'vnn_details' not in vnn_detail_for_re_embed :
                             vnn_detail_for_re_embed['vnn_details'] = {
                                 'cpu': n_alloc['cpu_reserved'], 
                                 'candidate_domains': ['ground', 'air', 'leo'] 
                             }
                        lost_nodes_this_vnr_on_sn_rem.append(vnn_detail_for_re_embed)
                if lost_nodes_this_vnr_on_sn_rem:
                    if vnr_id not in affected_vnr_details: affected_vnr_details[vnr_id]={'nodes':[],'links':list(allocs.get('links',[]))}
                    affected_vnr_details[vnr_id]['nodes'].extend(lost_nodes_this_vnr_on_sn_rem)

        for n_id in sorted_to_rem: 
            if n_id in self.network: self.network.remove_node(n_id)
        
        counts={'air':self.count_type('air'),'leo':self.count_type('leo')} # Keep full names for keys
        miss_a=self.air_nodes-counts['air']
        for _ in range(miss_a):
            nid=self._get_next_node_id(f"air_dyn_{self.added_counts['air']}")
            pos=(self.rng_np.uniform(0,100),self.rng_np.uniform(0,100),self.rng_np.uniform(100,500))
            cpu=self.rng_np.randint(50,101); lt=self.rng_np.exponential(scale=5400)
            self.initial_node_lifetimes["air"].append(lt)
            self.network.add_node(nid,type="air",cpu=cpu,initial_cpu=cpu,position=pos,lifetime=lt,initial_lifetime=lt)
            self.added_counts['air']+=1; self._last_added_nodes.append(nid); self._last_added_counts_this_step['air']+=1 # Use 'air'
        
        # Corrected key from 'l' to 'leo'
        miss_l=self.leo_nodes - counts.get('leo', 0) # Use .get for safety, though 'leo' should exist
        for _ in range(miss_l):
            nid=self._get_next_node_id(f"leo_dyn_{self.added_counts['leo']}")
            pos=(self.rng_np.uniform(0,100),self.rng_np.uniform(0,100),self.rng_np.uniform(500,2000))
            cpu=self.rng_np.randint(50,81); lt=self.rng_np.exponential(scale=600)
            self.initial_node_lifetimes["leo"].append(lt)
            self.network.add_node(nid,type="leo",cpu=cpu,initial_cpu=cpu,position=pos,lifetime=lt,initial_lifetime=lt)
            self.added_counts['leo']+=1; self._last_added_nodes.append(nid); self._last_added_counts_this_step['leo']+=1 # Use 'leo'
            
        if self._last_added_nodes: self._incremental_edge_update(); self._patch_isolated_nodes()    
        if self.network.number_of_nodes()>0 and not nx.is_connected(self.network): self._ensure_global_connectivity_after_dynamics()
        return affected_vnr_details 

    def get_last_removed_counts(self): return self._last_removed_counts_this_step.copy()
    def get_last_added_counts(self): return self._last_added_counts_this_step.copy()
    def get_all_initial_lifetimes(self): return {"air": list(self.initial_node_lifetimes["air"]), "leo": list(self.initial_node_lifetimes["leo"])}
    def _ensure_global_connectivity_after_dynamics(self):
        G=self.network
        if G.number_of_nodes()==0 or nx.is_connected(G): return
        comps=sorted(list(nx.connected_components(G)),key=lambda c:sorted(list(c))[0]); main_c=set(comps[0])
        for i in range(1,len(comps)):
            comp_s_set=set(comps[i]); main_s=sorted(list(main_c)); other_s=sorted(list(comp_s_set)); conn=False
            for u_n in main_s:
                for v_n in other_s:
                    type_u=G.nodes[u_n]['type']; type_v=G.nodes[v_n]['type']
                    if not ((type_u=='ground' and type_v=='leo')or(type_u=='leo' and type_v=='ground')): # Use full names
                        bw=self.rng_np.randint(50,81) if (type_u=='air' and type_v=='leo')or(type_u=='leo' and type_v=='air') else self.rng_np.randint(50,101)
                        G.add_edge(u_n,v_n,bandwidth=bw,initial_bandwidth=bw);conn=True;break
                if conn: break
            if not conn and main_s and other_s:
                uf,vf=self.rng_py.choice(main_s),self.rng_py.choice(other_s)
                bw=self.rng_np.randint(50,101);G.add_edge(uf,vf,bandwidth=bw,initial_bandwidth=bw)
            main_c.update(comp_s_set)

    def get_node_utilization(self) -> float:
        init_cpu=sum(d.get('initial_cpu',0) for _,d in self.network.nodes(data=True))
        free_cpu=sum(d.get('cpu',0) for _,d in self.network.nodes(data=True))
        return (init_cpu-free_cpu)/init_cpu if init_cpu!=0 else 0.0
    def get_link_utilization(self) -> float:
        init_bw=sum(d.get('initial_bandwidth',0) for _,_,d in self.network.edges(data=True))
        free_bw=sum(d.get('bandwidth',0) for _,_,d in self.network.edges(data=True))
        return (init_bw-free_bw)/init_bw if init_bw!=0 else 0.0
    def get_overall_utilization(self) -> float:
        init_cpu_total = sum(d.get('initial_cpu',0) for _,d in self.network.nodes(data=True))
        used_cpu_total = init_cpu_total - sum(d.get('cpu',0) for _,d in self.network.nodes(data=True))
        init_bw_total = sum(d.get('initial_bandwidth',0) for _,_,d in self.network.edges(data=True))
        used_bw_total = init_bw_total - sum(d.get('bandwidth',0) for _,_,d in self.network.edges(data=True))
        total_initial_capacity = init_cpu_total + init_bw_total
        total_used_capacity = used_cpu_total + used_bw_total
        return total_used_capacity / total_initial_capacity if total_initial_capacity > 0 else 0.0
    def count_type(self,n_type): return sum(1 for _,d in self.network.nodes(data=True) if d['type']==n_type)
    def get_available_cpu(self,n_id): return self.network.nodes[n_id].get('cpu',0) if n_id in self.network.nodes else 0
    def get_total_adjacent_bandwidth(self,n_id):
        bw=0
        if n_id in self.network:
            for neigh in sorted(list(self.network.neighbors(n_id))):
                if self.network.has_edge(n_id,neigh): bw+=self.network.edges[n_id,neigh].get('bandwidth',0)
        return bw
