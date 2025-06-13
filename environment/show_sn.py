import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Make sure the latest version of your SubstrateNetwork class is in a file named
# substrate_network.py in the same directory as this script.
# This script assumes you are using the version with internal RNGs
# (ID: substrate_network_deterministic_rng).
from substrate_network import SubstrateNetwork

# --- Configuration ---
SIMULATION_STEPS = 5000
NETWORK_SEED = 42  # Use a fixed seed for reproducible visualizations

def plot_network(G, ax, title):
    """
    Helper function to plot the substrate network graph on a given matplotlib axis
    with a clear layered layout for different domains.
    """
    # Get node types for coloring and positioning
    node_types = {node: data['type'] for node, data in G.nodes(data=True)}

    # Define colors for each domain
    color_map = {
        'ground': '#4CAF50',  # Green
        'air': '#2196F3',     # Blue
        'leo': '#FFC107'      # Amber
    }
    
    # Assign a color to each node based on its type
    node_colors = [color_map.get(node_types.get(node, ''), '#CCCCCC') for node in G.nodes()]

    # --- Layered Layout Position Calculation ---
    pos = {}
    # Separate nodes by domain/layer
    ground_nodes = sorted([n for n, t in node_types.items() if t == 'ground'])
    air_nodes = sorted([n for n, t in node_types.items() if t == 'air'])
    leo_nodes = sorted([n for n, t in node_types.items() if t == 'leo'])

    # Function to spread nodes horizontally within a layer
    def get_x_positions(nodes):
        return np.linspace(0, 100, len(nodes)) if len(nodes) > 0 else []

    # Assign positions for each layer
    for node, x_pos in zip(ground_nodes, get_x_positions(ground_nodes)):
        pos[node] = (x_pos, 1) # Ground layer at y=1
    for node, x_pos in zip(air_nodes, get_x_positions(air_nodes)):
        pos[node] = (x_pos, 2) # Air layer at y=2
    for node, x_pos in zip(leo_nodes, get_x_positions(leo_nodes)):
        pos[node] = (x_pos, 3) # LEO/Space layer at y=3

    # Draw the network using the new layered positions
    nx.draw(
        G, 
        pos, 
        ax=ax,
        with_labels=False, 
        node_size=50,
        node_color=node_colors,
        width=0.5,
        edge_color='#AAAAAA'
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='y', left=True, labelleft=True)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Ground', 'Air', 'LEO/Space'])


def visualize_network_evolution():
    """
    Main function to initialize, evolve, and plot the substrate network.
    """
    print(f"Initializing Substrate Network with seed {NETWORK_SEED}...")
    sn = SubstrateNetwork(seed=NETWORK_SEED)

    # --- Plot Initial State (t=0) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Made figure wider
    fig.suptitle('Substrate Network Topology Change (Layered Layout)', fontsize=18)
    
    # Create a copy of the graph at t=0 to plot
    initial_graph = sn.network.copy()
    plot_network(initial_graph, axes[0], f"Initial State (t=0)")
    print(f"Plotted initial network with {initial_graph.number_of_nodes()} nodes and {initial_graph.number_of_edges()} edges.")

    # --- Run Simulation to Evolve the Network ---
    print(f"Running simulation for {SIMULATION_STEPS} timesteps...")
    for i in range(SIMULATION_STEPS):
        if (i + 1) % 500 == 0: # Print progress
            print(f"  ...at timestep {i+1}/{SIMULATION_STEPS}")
        sn.step_dynamics()
    print("Simulation finished.")

    # --- Plot Final State (t=5000) ---
    final_graph = sn.network.copy()
    plot_network(final_graph, axes[1], f"Final State (t={SIMULATION_STEPS})")
    print(f"Plotted final network with {final_graph.number_of_nodes()} nodes and {final_graph.number_of_edges()} edges.")

    # --- Create a legend ---
    # Legend is now less critical as y-axis is labeled, but can be kept for color key
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Ground SNN', markersize=10, markerfacecolor='#4CAF50'),
        plt.Line2D([0], [0], marker='o', color='w', label='Air SNN', markersize=10, markerfacecolor='#2196F3'),
        plt.Line2D([0], [0], marker='o', color='w', label='LEO SNN', markersize=10, markerfacecolor='#FFC107')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for legend and title
    
    # Save and show the plot
    plot_filename = "substrate_network_evolution_layered.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as {plot_filename}")
    plt.show()

if __name__ == '__main__':
    visualize_network_evolution()
