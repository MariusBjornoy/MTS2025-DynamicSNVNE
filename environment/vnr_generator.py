import uuid
import numpy as np
import random

# Initialize these at the module level (outside any function)
VNR_GEN_PRINT_COUNT = 0
MAX_VNR_GEN_PRINTS = 5 # For limiting debug prints if uncommented

def generate_vnr(seed_counter=None, ttl=250): 
    global VNR_GEN_PRINT_COUNT 

    # Determine VNR ID
    if seed_counter is not None:
        vnr_id = f"vnr_{seed_counter}"
    else:
        vnr_id = f"vnr_{uuid.uuid4().hex[:8]}" # Shorter UUID for readability

    # VNR Lifetime (TTL) - sampled from exponential distribution
    # The 'ttl' parameter is the mean (scale) of this distribution.
    ttl_val = np.random.exponential(scale=ttl) 
    if ttl_val < 1: ttl_val = 1 # Ensure a minimum TTL

    # Number of Virtual Network Nodes (VNNs)
    num_vnns = np.random.randint(2, 7) # Generates VNRs with 2 to 6 nodes

    # Generate VNNs
    possible_domains = ['ground', 'air', 'leo']
    vnns = [] # This list will store VNN objects in their creation order (vnn_0, vnn_1, etc.)
    for i in range(num_vnns):
        cpu_req = np.random.randint(1, 21) # CPU requirement: 1 to 20
        k_domains = np.random.randint(1, len(possible_domains) + 1) # Number of candidate domains (1 to 3)
        candidate_domains_for_vnn = random.sample(possible_domains, k_domains)
        vnns.append({
            'id': f'{vnr_id}_vnn_{i}', # More unique VNN ID
            'cpu': cpu_req,
            'candidate_domains': candidate_domains_for_vnn,
            'parent_vnr': vnr_id # Link back to parent VNR
        })

    links = []
    vnn_ids_all = [v['id'] for v in vnns] # All VNN IDs in their creation order
    
    # Establish a random topological order (path order) for DAG generation
    # This uses the global `random` which should be seeded by the calling script.
    path_ordered_vnn_ids = random.sample(vnn_ids_all, len(vnn_ids_all))

    # For a strictly linear graph, the number of links is num_vnns - 1
    if num_vnns > 1:
        num_total_links = num_vnns - 1
        
        # Create a directed path along the path_ordered_vnn_ids
        for i in range(num_total_links): # This will iterate num_vnns - 1 times
            source_node_id = path_ordered_vnn_ids[i]
            target_node_id = path_ordered_vnn_ids[i+1]
            
            bw_req = np.random.randint(1, 51) # Bandwidth requirement: 1 to 50
            links.append({
                'id': f'{vnr_id}_vlink_{len(links)}',
                'from': source_node_id, 
                'to': target_node_id,   
                'bandwidth': bw_req
            })
    else: # num_vnns is 0 or 1
        num_total_links = 0
        path_ordered_vnn_ids = [] # Ensure it's an empty list if no nodes or one node
    
    return {
        'id': vnr_id,
        'ttl': ttl_val, 
        'nodes': vnns, # List of VNN dicts (in vnn_0, vnn_1 order)
        'links': links, 
        'path_ordered_vnn_ids': path_ordered_vnn_ids, # <<<--- ADDED THIS KEY
        # 'ordered_nodes' and 'ordered_links' were just copies, can be removed if path_ordered_vnn_ids is used
    }

def generate_vnrs_seeded(n, seed=None, mean_ttl=250): 
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return [generate_vnr(seed_counter=i, ttl=mean_ttl) for i in range(n)]

# Example usage:
if __name__ == '__main__':
    np.random.seed(42); random.seed(42)
    print("Generating 2 sample VNRs (Strictly Linear DAGs) with mean TTL of 200:")
    for i in range(2):
        vnr = generate_vnr(seed_counter=i, ttl=200)
        print(f"\nVNR ID: {vnr['id']}, TTL: {vnr['ttl']:.2f}, Nodes: {len(vnr['nodes'])}, Links: {len(vnr['links'])}")
        print(f"  Path Ordered VNN IDs: {vnr['path_ordered_vnn_ids']}")
        print(f"  Nodes (generation order):")
        for node in vnr['nodes']:
            print(f"    {node['id']}: CPU {node['cpu']}, Domains {node['candidate_domains']}")
        print(f"  Links ({len(vnr['links'])}):")
        for link in vnr['links']:
            print(f"    {link['id']}: {link['from']} -> {link['to']}, BW {link['bandwidth']}")
