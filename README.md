# MTS2025

Abstract:
This thesis addresses the problem of dynamic network service allocation within Space-Air-Ground
Integrated Networks (SAGIN), a key component of future 6G mobile communication. The inherent
dynamics of SAGIN, characterized by frequent changes in topology due to the mobility of aerial
and satellite nodes, pose significant challenges for traditional Virtual Network Embedding (VNE)
algorithms. To address these challenges, a resource-aware solution based on Deep Reinforcement
Learning (DRL) was developed.

A simulation environment was designed to model a multi-domain SAGIN, featuring a dynamic
Substrate Network (SN) where nodes in the air and space layers with finite lifetimes based on
realistic reachability windows. The arrival of Virtual Network Requests (VNRs) is modeled as
a Poisson process. A DRL agent that uses , employing the Proximal Policy Optimization (PPO)
algorithm is implemented and trained to solve the VNE problem by making node mapping decisions
in this constantly changing environment. The agent’s performance is rigorously evaluated based on
key metrics, including the VNR acceptance rate, revenue-to-cost ratio, and its ability to handle the
network disruptions.

The results demonstrate that the DRL-based approach can achieve a high VNR acceptance rate
of approximately 80% under baseline conditions, validating its effectiveness in a dynamic setting.
The research highlights the significant impact of network dynamicity, with nearly 16% of initially
embedded VNRs being disrupted during their lifetime. This work underscores the need for adaptive,
learning-based approaches for resource management in the complex and evolving landscape of
SAGINs.

## Results
The figures generated, with the appropriate JSON files are in the folders:
- /50_ttl_ppo_vne_parameter_sweep_results -> For the appendix results with TTL iterations of 100-400 with 50 in between
- /Final_arrival -> Displays the plots of the arrival section
- /Final_ttl -> Displays the plots of the TTL section

## Trained PPO Agent
The final Trained PPO agent is available in the folder:
* /five_model
This is the agent used for all testing


# Reinforcement Learning for Dynamic Virtual Network Embedding in Multi-Domain Networks

This repository contains the source code for a Master's thesis project on Virtual Network Embedding (VNE) using Reinforcement Learning (RL). The system is designed to embed Virtual Network Requests (VNRs) onto a dynamic, multi-domain (Ground, Air, Space) substrate network. The solution employs a Proximal Policy Optimization (PPO) agent, implemented using the Stable Baselines 3 library.

## System Architecture

The project is built around three main components that simulate the VNE problem within a Reinforcement Learning framework:

1.  **`SubstrateNetwork`**: This class (`substrate_network.py`) models the physical network infrastructure. It's a dynamic graph with nodes across three domains (Ground, Air, Space). Air and Space nodes have limited lifetimes and are periodically replaced, simulating real-world network churn.
2.  **`VNEEnv`**: This class (`vne_env.py`) creates a custom environment that adheres to the Gymnasium (formerly OpenAI Gym) API. It acts as the bridge between the RL agent and the VNE problem. It manages the state representation, action execution (placing virtual nodes), reward calculation, and the overall lifecycle of a VNR embedding episode.
3.  **PPO Agent**: The "brain" of the operation. The agent is trained using `train.py` to learn a policy for making optimal node placement decisions. It takes observations from the `VNEEnv` and outputs an action, which corresponds to selecting a substrate node for a virtual node.

## Prerequisites & Setup

This project is written in Python 3. The required libraries can be installed using pip.

```bash
pip install gymnasium numpy networkx matplotlib stable-baselines3

    gymnasium: The toolkit for creating and interacting with the RL environment.

    numpy: For numerical operations.

    networkx: For creating and managing the graph structures of the substrate and virtual networks.

    matplotlib: For generating plots to visualize results.

    stable-baselines3: The RL library used to implement and train the PPO agent.

File Breakdown

The repository is structured into core components, training/simulation scripts, and analysis tools.
Core Components

    substrate_network.py

        Purpose: Defines the SubstrateNetwork class, which simulates the physical network. It handles the creation of the multi-domain network, dynamic events like node failures and additions, and manages resource allocation (CPU, bandwidth).

        Key Components: SubstrateNetwork class, step_dynamics() method for network evolution, embed_vnn() and embed_vnl() for resource commitment.

    vnr_generator.py

        Purpose: Generates Virtual Network Requests (VNRs). Each VNR is a small graph with CPU and bandwidth requirements for its nodes and links, respectively, along with a lifetime (TTL).

        Key Components: generate_vnr() function.

    vne_env.py

        Purpose: Implements the VNEEnv class, the custom Gymnasium environment for the VNE problem. It defines the state-action-reward loop for the RL agent.

        Key Components: VNEEnv class, step() and reset() methods, observation space definition, and reward calculation logic.

        Note: vneEnv.py appears to be an older or duplicate version. vne_env.py is the version used in the primary scripts.

Training & Simulation

    train.py

        Purpose: To train the PPO agent. This script initializes the VNEEnv, configures the PPO model with specific hyperparameters, and runs the training loop for a defined number of timesteps.

        Usage: Run from the command line. The script will save TensorBoard logs and model checkpoints to the five_model/ directory.

    python train.py

    simulation.py

        Purpose: To evaluate a pre-trained agent. It loads a saved model, runs it on a series of VNRs in a simulation environment, and collects detailed performance metrics.

        Usage: Before running, you must update the MODEL_PATH variable to point to your trained model (e.g., five_model/models/.../final_model.zip). The results are saved as a detailed JSON file in the ppo_vne_simulation_results/ directory.

    python simulation.py

    multiple_params.py

        Purpose: An advanced simulation script that runs a parameter sweep. It tests the agent's performance across a grid of different VNR arrival rates and mean TTLs.

        Usage: Configure the VNR_INTER_ARRIVAL_MEAN_LIST and VNR_MEAN_TTL_LIST variables in the script. It will run a full simulation for each parameter combination and save the results to the directory specified by RESULTS_DIR.

    python multiple_params.py

Plotting & Analysis

    show_sn.py

        Purpose: A utility script to visualize the substrate network's topology. It generates a plot showing the network's initial state and its state after a set number of dynamic steps, providing insight into the network's evolution.

        Usage:

    python show_sn.py

    plot_simulation_results.py

        Purpose: To visualize the data from a single simulation run. It takes the JSON file generated by simulation.py and creates a folder of plots showing metrics like VNR concurrency, acceptance rate over time, resource cost distribution, etc.

        Usage: Pass the path to a results JSON file as a command-line argument. If no path is given, it will try to find the most recent results file in ppo_vne_simulation_results/.

    python plot_simulation_results.py "path/to/your/sim_results.json"

    plot_multiple.py

        Purpose: To visualize the data from a parameter sweep. It reads all JSON files from the results directory (configured via the RESULTS_DIR variable, e.g., "Final_ttl/") and generates a series of grid plots comparing performance across the different parameters. It also creates a parameter_sweep_summary.csv file.

        Usage:

    python plot_multiple.py

Workflow Guide

Follow these steps to train an agent and evaluate its performance.
Step 1: Train the RL Agent

Run the training script. This may take a significant amount of time. Monitor the progress in your console or via TensorBoard.

# Start the training process
python train.py

# (Optional) View live training metrics with TensorBoard
# tensorboard --logdir five_model/tensorboard_logs

This will create a trained model file, typically named final_model.zip, inside a subdirectory within five_model/models/.
Step 2: Run a Simulation

Once you have a trained model, edit simulation.py to set the MODEL_PATH variable to the path of your final_model.zip. Then, run the simulation.

# Example: MODEL_PATH = "five_model/models/ppo_vne_..._5Msteps/final_model.zip"
python simulation.py

This will produce a JSON file (e.g., sim_results_...json) in the ppo_vne_simulation_results/ directory.
Step 3: Run a Parameter Sweep

For a more thorough evaluation, edit multiple_params.py to set the MODEL_PATH and configure the parameter grids (VNR_INTER_ARRIVAL_MEAN_LIST, VNR_MEAN_TTL_LIST). Then run the script.

python multiple_params.py

This will generate multiple JSON files in the specified RESULTS_DIR.
Step 4: Visualize Results

Use the plotting scripts to analyze the JSON files generated in the previous steps.

    For a single simulation run:

    python plot_simulation_results.py "ppo_vne_simulation_results/your_results_file.json"

    For a parameter sweep:

    # Ensure the RESULTS_DIR in plot_multiple.py matches the one in multiple_params.py
    python plot_multiple.py





