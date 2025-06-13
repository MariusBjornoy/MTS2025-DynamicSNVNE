import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Font Size Configuration for Reports ---
# These variables control the font sizes for various elements in the plots.
# Feel free to adjust these values to fit your paper's style guidelines.
TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 16
BAR_LABEL_FONTSIZE = 12
PIE_CHART_TEXT_FONTSIZE = 14

# Global variable for RE_EMBED_BROKEN_VNRS to be accessed by plotting functions
# This will be updated from the JSON configuration when main() is called.
RE_EMBED_BROKEN_VNRS = False 

def plot_lifetime_distribution(lifetimes, domain_name, save_path):
    if not lifetimes: print(f"No lifetime data for {domain_name} nodes to plot."); return
    plt.figure(figsize=(10, 6))
    plt.hist(lifetimes, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Lifetimes for {domain_name} Nodes', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Lifetime (simulation time steps) [S]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Frequency (Number of Nodes)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{domain_name.lower()}_all_generated_lifetime_distribution.png"))
    plt.close()
    print(f"All generated {domain_name} lifetime distribution plot saved to {save_path}")

def plot_concurrency(concurrency_data, save_path):
    if not concurrency_data: print("No concurrency data to plot."); return
    times = [dp['time'] for dp in concurrency_data]
    counts = [dp['active_vnrs'] for dp in concurrency_data]
    plt.figure(figsize=(12, 6))
    plt.plot(times, counts, linestyle='-', marker='.', markersize=3)
    plt.title('VNR Concurrency Over Simulation Time', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Simulation Time Step [S]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Number of Active VNRs', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "vnr_concurrency.png"))
    plt.close()
    print(f"VNR concurrency plot saved to {save_path}")

def plot_cost_revenue_per_embed(vnr_details, metric_key, title_prefix, save_path):
    values = [vnr[metric_key] for vnr in vnr_details if vnr.get('status') == 'successfully_embedded' and metric_key in vnr and vnr[metric_key] is not None]
    if not values: print(f"No {metric_key} data for successful embeds to plot."); return
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, edgecolor='black', alpha=0.7, color='skyblue' if 'revenue' in metric_key else 'salmon')
    plt.title(f'Distribution of {title_prefix} per Successful VNR', fontsize=TITLE_FONTSIZE)
    plt.xlabel(title_prefix, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric_key.lower()}_per_embed_distribution.png"))
    plt.close()
    print(f"{title_prefix} per embed distribution plot saved to {save_path}")

def plot_embedding_events(events_summary, final_summary, save_path):
    global RE_EMBED_BROKEN_VNRS 
    if not events_summary and not final_summary: print("No embedding events summary to plot."); return
    total_embedded_successfully = final_summary.get('total_vnrs_embedded_successfully', 0)
    total_failed_initial_embedding = final_summary.get('total_vnrs_failed_initial_embedding', 0)
    total_broken_by_dynamics = events_summary.get('total_vnrs_broken_by_dynamics', 0)
    total_vnrs_re_embedded_successfully = events_summary.get('total_vnrs_node_re_embedded_successfully', 0) 

    labels = ['Initial Success', 'Initial Fail']
    counts = [total_embedded_successfully, total_failed_initial_embedding]
    colors = ['green', 'red']
    if RE_EMBED_BROKEN_VNRS: 
        labels.extend(['Broken by Dynamics', 'VNRs Re-embedded (Nodes)']) 
        counts.extend([total_broken_by_dynamics, total_vnrs_re_embedded_successfully]) 
        colors.extend(['orange', 'cyan'])
    plotted_labels = []; plotted_counts = []; plotted_colors = []
    for i, label in enumerate(labels):
        if counts[i] > 0 or (RE_EMBED_BROKEN_VNRS and ("Broken" in label or "Re-embedded" in label)):
            plotted_labels.append(label); plotted_counts.append(counts[i]); plotted_colors.append(colors[i])
    if not plotted_counts or all(c == 0 for c in plotted_counts): print("No non-zero embedding events to plot after filtering."); return
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(plotted_labels, plotted_counts, color=plotted_colors)
    plt.title('Summary of VNR Embedding Lifecycle Events', fontsize=TITLE_FONTSIZE)
    plt.ylabel('Number of VNRs', fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(rotation=15, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    plt.tick_params(axis='y', which='major', labelsize=TICK_LABEL_FONTSIZE)
    for bar in bars: 
        yval = bar.get_height()
        if yval > 0 : plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(plotted_counts)*0.01 if plotted_counts and max(plotted_counts)>0 else 0.1), int(yval), ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "embedding_events_summary.png"))
    plt.close()
    print(f"Embedding events summary plot saved to {save_path}")

def plot_broken_vnr_acceptance_rate(events_summary, save_path): 
    global RE_EMBED_BROKEN_VNRS 
    if not RE_EMBED_BROKEN_VNRS: print("Re-embedding disabled; skipping broken VNR re-embedding acceptance rate plot."); return
    total_broken_vnrs = events_summary.get('total_vnrs_broken_by_dynamics', 0)
    total_vnrs_re_embedded_success = events_summary.get('total_vnrs_node_re_embedded_successfully', 0) 
    if total_broken_vnrs == 0:
        print("No VNRs were broken by dynamics; skipping broken VNR re-embedding acceptance rate plot.")
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, "No VNRs were broken by dynamics.", horizontalalignment='center', verticalalignment='center', fontsize=AXIS_LABEL_FONTSIZE)
        plt.title('Re-embedding Acceptance Rate for Broken VNRs', fontsize=TITLE_FONTSIZE)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, "broken_vnr_re_embedding_acceptance_rate.png"))
        plt.close()
        print(f"Broken VNR re-embedding acceptance rate plot (no data) saved to {save_path}"); return
    acceptance_rate = total_vnrs_re_embedded_success / total_broken_vnrs if total_broken_vnrs > 0 else 0
    failed_re_embedding_vnrs = total_broken_vnrs - total_vnrs_re_embedded_success
    labels = [f'Success ({total_vnrs_re_embedded_success})', f'Fail ({failed_re_embedding_vnrs})']
    sizes = [max(0, s) for s in [total_vnrs_re_embedded_success, failed_re_embedding_vnrs]]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0) if total_vnrs_re_embedded_success > 0 and sum(sizes) > 0 else (0,0) 
    if sum(sizes) > 0:
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': PIE_CHART_TEXT_FONTSIZE})
        plt.title(f'Re-embedding Success Rate for Broken VNRs\n(Total Broken: {total_broken_vnrs})', fontsize=TITLE_FONTSIZE)
    else: print("No data for broken VNR re-embedding pie chart (all counts zero after processing)."); return 
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "broken_vnr_re_embedding_acceptance_rate.png"))
    plt.close()
    print(f"Broken VNR re-embedding acceptance rate plot saved to {save_path}")

def plot_cumulative_snn_stats(snn_data, data_type, save_path): 
    if not snn_data: print(f"No SNN {data_type} data to plot."); return
    times = [dp['time'] for dp in snn_data]
    ground_stats = [dp['ground'] for dp in snn_data]
    air_stats = [dp['air'] for dp in snn_data]
    leo_stats = [dp['leo'] for dp in snn_data]
    plt.figure(figsize=(12, 6))
    if not all(g == 0 for g in ground_stats): plt.plot(times, ground_stats, label=f'Ground Nodes {data_type.capitalize()}', marker='.')
    else: print(f"Ground node {data_type} count is zero throughout, not plotted for SNN {data_type} stats.")
    plt.plot(times, air_stats, label=f'Air Nodes {data_type.capitalize()}', marker='.')
    plt.plot(times, leo_stats, label=f'LEO Nodes {data_type.capitalize()}', marker='.')
    plt.title(f'Cumulative Substrate Nodes {data_type.capitalize()} Over Time', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Simulation Time Step [S]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(f'Cumulative Count of Nodes', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"cumulative_snn_{data_type}.png"))
    plt.close()
    print(f"Cumulative SNN {data_type} plot saved to {save_path}")

def plot_vnr_attribute_distribution(vnr_details, attribute_key, title, xlabel, save_path, bins=20):
    values = [vnr[attribute_key] for vnr in vnr_details if attribute_key in vnr and vnr[attribute_key] is not None]
    if not values: print(f"No data for VNR attribute '{attribute_key}' to plot."); return
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if not numeric_values: print(f"No numeric data for VNR attribute '{attribute_key}' to plot after filtering."); return
    plt.figure(figsize=(10, 6))
    plt.hist(numeric_values, bins=bins, edgecolor='black', alpha=0.7, color='dodgerblue')
    plt.title(f'Distribution of VNR {title}', fontsize=TITLE_FONTSIZE)
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"vnr_{attribute_key.replace('/','_')}_distribution.png"))
    plt.close()
    print(f"VNR {title} distribution plot saved to {save_path}")

def plot_cumulative_broken_vnrs(broken_vnr_data, save_path):
    if not broken_vnr_data: print("No cumulative broken VNR data to plot."); return
    times = [dp['time'] for dp in broken_vnr_data]
    cumulative_counts = [dp['cumulative_broken_vnrs'] for dp in broken_vnr_data]
    plt.figure(figsize=(12, 6))
    plt.plot(times, cumulative_counts, linestyle='-', marker='.', color='sienna', markersize=3)
    plt.title('Cumulative VNRs Broken by Dynamics Over Time', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Simulation Time Step [S]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Cumulative Count of Broken VNRs', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cumulative_broken_vnrs.png"))
    plt.close()
    print(f"Cumulative broken VNRs plot saved to {save_path}")

def plot_per_vnr_acceptance_rate(vnr_details, save_path):
    if not vnr_details: print("No VNR details to plot per-VNR acceptance rate."); return
    successful_embeddings = 0; processed_vnrs_count = 0
    acceptance_rate_progression = []; vnr_indices = []
    for i, vnr in enumerate(vnr_details):
        processed_vnrs_count += 1
        if vnr.get('status') == 'successfully_embedded': successful_embeddings += 1
        acceptance_rate_progression.append(successful_embeddings / processed_vnrs_count)
        vnr_indices.append(processed_vnrs_count)
    if not vnr_indices: print("No VNRs processed to plot detailed acceptance rate."); return
    plt.figure(figsize=(12, 6))
    plt.plot(vnr_indices, acceptance_rate_progression, linestyle='-', marker=None, color='teal',linewidth = 2) 
    plt.title('Acceptance Rate Progression', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Number of VNRs Processed', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Cumulative Acceptance Rate', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.ylim(0, 1.05)
    plt.xlim(0, max(vnr_indices) + 1 if vnr_indices else 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "per_vnr_acceptance_rate_progression.png"))
    plt.close()
    print(f"Per-VNR acceptance rate progression plot saved to {save_path}")

def plot_per_vnr_rc_ratio(vnr_details, save_path):
    if not vnr_details: print("No VNR details to plot R/C ratio per VNR."); return
    vnr_indices = []
    rc_ratios = []

    for i, vnr in enumerate(vnr_details):
        if vnr.get('status') == 'successfully_embedded':
            revenue = vnr.get('revenue', 0.0); cost = vnr.get('cost', 0.0)
            ratio = revenue / cost if cost > 0 else 5.0 if revenue > 0 else 1.0 
            vnr_indices.append(i + 1); rc_ratios.append(ratio)

    if not vnr_indices: print("No successfully embedded VNRs to plot R/C ratio for."); return

    plt.figure(figsize=(12, 6))
    plt.scatter(vnr_indices, rc_ratios, marker='o', s=5, alpha=0.5, color='blue', label='R/C Ratio of Successful VNR')
    
    if len(vnr_indices) > 20:
        window_size = 20
        moving_avg = np.convolve(rc_ratios, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = vnr_indices[window_size-1:]
        plt.plot(moving_avg_x, moving_avg, color='green', linewidth=2, label=f'{window_size}-Success Moving Avg')

    plt.title('Revenue-to-Cost Ratio per Successful Embedding', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Number of VNRs Processed', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Individual Revenue / Cost Ratio', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.xlim(0, len(vnr_details) + 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "per_vnr_rc_ratio_event.png"))
    plt.close()
    print(f"Per-VNR revenue/cost ratio event plot saved to {save_path}")

def plot_vnr_rc_ratio(vnr_details, save_path): 
    if not vnr_details: print("No VNR details to plot R/C ratio per VNR."); return
    vnr_indices = []; rc_ratios = []

    for i, vnr in enumerate(vnr_details):
        if vnr.get('status') == 'successfully_embedded':
            revenue = vnr.get('revenue', 0.0); cost = vnr.get('cost', 0.0)
            ratio = revenue / cost if cost > 0 else 5.0 if revenue > 0 else 1.0
            vnr_indices.append(i + 1); rc_ratios.append(ratio)

    if not vnr_indices: print("No successfully embedded VNRs to plot R/C ratio for."); return

    plt.figure(figsize=(12, 6))
    if len(vnr_indices) > 20:
        window_size = 20
        moving_avg = np.convolve(rc_ratios, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = vnr_indices[window_size-1:]
        plt.plot(moving_avg_x, moving_avg, color='green', linewidth=2, label=f'{window_size}-Success Moving Avg')

    plt.title('Revenue-to-Cost Ratio per Successful VNR Embedding', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Number of VNRs Processed', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Individual Revenue / Cost Ratio', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.xlim(0, len(vnr_details) + 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "RC_graph.png"))
    plt.close()
    print(f"Per-VNR revenue/cost ratio event plot saved to {save_path}")

def plot_embedding_runtime(vnr_details, save_path):
    """ Plots the runtime for each VNR embedding episode. """
    if not vnr_details: print("No VNR details to plot embedding runtime."); return

    runtimes_ms = []; vnr_indices_with_data = []
    for i, vnr in enumerate(vnr_details):
        if 'embedding_time_ms' in vnr and vnr['embedding_time_ms'] is not None:
            runtimes_ms.append(vnr['embedding_time_ms'])
            vnr_indices_with_data.append(i + 1)
    
    if not runtimes_ms: print("No embedding runtime data found in VNR details."); return

    overall_average_runtime = np.mean(runtimes_ms)
    cumulative_average_runtime = np.cumsum(runtimes_ms) / np.arange(1, len(runtimes_ms) + 1)

    plt.figure(figsize=(12, 7))
    
    plt.scatter(vnr_indices_with_data, runtimes_ms, marker='.', alpha=0.3, color='gray', label='Individual Time')
    plt.plot(vnr_indices_with_data, cumulative_average_runtime, color='b', linestyle='-', linewidth=2, label=f'Cumulative Average')
    plt.axhline(y=overall_average_runtime, color='r', linestyle='--', linewidth=2, label=f'Overall Average: {overall_average_runtime:.2f} ms')
    
    plt.title('VNR Embedding Runtime per Episode', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Number of VNRs Processed', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Runtime (ms)', fontsize=AXIS_LABEL_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    
    if len(runtimes_ms) > 1:
        p99 = np.percentile(runtimes_ms, 99)
        plt.ylim(0, p99 * 1.2)
    
    plot_filename = os.path.join(save_path, "vnr_embedding_runtime.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"VNR embedding runtime plot saved to {plot_filename}")


def main(results_filepath):
    global RE_EMBED_BROKEN_VNRS 
    if not os.path.exists(results_filepath): print(f"Error: Results file not found at {results_filepath}"); return
    with open(results_filepath, 'r') as f: results_data = json.load(f)
    base_results_filename = os.path.splitext(os.path.basename(results_filepath))[0]
    plot_subdir = os.path.join(os.path.dirname(results_filepath), base_results_filename + "_plots")
    os.makedirs(plot_subdir, exist_ok=True)
    print(f"\n--- Loading results from: {results_filepath} ---")
    print(f"--- Saving plots to: {plot_subdir} ---")
    RE_EMBED_BROKEN_VNRS = results_data.get("configuration", {}).get("re_embed_broken_vnrs", False)
    print(f"Plotting with RE_EMBED_BROKEN_VNRS set to: {RE_EMBED_BROKEN_VNRS}")

    plot_lifetime_distribution(results_data.get("initial_lifetimes", {}).get("air", []), "Air", plot_subdir)
    plot_lifetime_distribution(results_data.get("initial_lifetimes", {}).get("leo", []), "LEO", plot_subdir)
    plot_concurrency(results_data.get("concurrency_data", []), plot_subdir)
    plot_cost_revenue_per_embed(results_data.get("vnr_details", []), 'cost', 'Cost', plot_subdir)
    plot_cost_revenue_per_embed(results_data.get("vnr_details", []), 'revenue', 'Revenue', plot_subdir)
    plot_embedding_events(results_data.get("simulation_events", {}), results_data.get("final_summary",{}), plot_subdir)
    plot_broken_vnr_acceptance_rate(results_data.get("simulation_events", {}), plot_subdir) 
    plot_cumulative_snn_stats(results_data.get("snn_removal_data", []), "removed", plot_subdir)
    plot_cumulative_snn_stats(results_data.get("snn_appearance_data", []), "added", plot_subdir) 
    plot_cumulative_broken_vnrs(results_data.get("broken_vnr_progression_data", []), plot_subdir) 
    plot_vnr_attribute_distribution(results_data.get("vnr_details", []), 'num_nodes', 'Length (Number of Nodes)', 'Number of VNNs', plot_subdir)
    plot_vnr_attribute_distribution(results_data.get("vnr_details", []), 'ttl_actual_sampled', 'TTL (Actual Sampled)', 'TTL Value', plot_subdir)
    
    plot_per_vnr_acceptance_rate(results_data.get("vnr_details", []), plot_subdir)
    plot_per_vnr_rc_ratio(results_data.get("vnr_details", []), plot_subdir) 
    plot_vnr_rc_ratio(results_data.get("vnr_details", []), plot_subdir) 
    plot_embedding_runtime(results_data.get("vnr_details", []), plot_subdir)

    if results_data.get("plot_data_points"): 
        plot_data = results_data["plot_data_points"]
        if plot_data: 
            vnrs_processed_agg = [dp['vnrs_processed'] for dp in plot_data] 
            acceptance_rates_agg = [dp['acceptance_rate'] for dp in plot_data] 
            rc_ratios_agg = [dp['revenue_to_cost_ratio'] for dp in plot_data] 
            fig, ax1 = plt.subplots(figsize=(12,6))
            color = 'tab:blue'
            ax1.set_xlabel('VNRs Processed (Sampled)', fontsize=AXIS_LABEL_FONTSIZE) 
            ax1.set_ylabel('Acceptance Rate', color=color, fontsize=AXIS_LABEL_FONTSIZE) 
            ax1.plot(vnrs_processed_agg, acceptance_rates_agg, color=color, marker='.', linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color, labelsize=TICK_LABEL_FONTSIZE)
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.set_ylim(0, 1.05)
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Overall Revenue/Cost Ratio', color=color, fontsize=AXIS_LABEL_FONTSIZE) 
            ax2.plot(vnrs_processed_agg, rc_ratios_agg, color=color, marker='x', linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=TICK_LABEL_FONTSIZE)
            fig.tight_layout()
            plt.title('Overall Acceptance Rate and R/C Ratio (Sampled)', fontsize=TITLE_FONTSIZE) 
            plt.savefig(os.path.join(plot_subdir, "overall_sampled_acceptance_rc_ratio_vs_vnrs.png")) 
            plt.close(fig)
            print(f"Overall sampled Acceptance Rate and R/C ratio plot saved to {plot_subdir}")
        else: print("No aggregated data points found for 'overall_sampled_acceptance_rc_ratio_vs_vnrs' plot.")
    print("\n--- All plots generated. ---")

    print("\n--- Additional Simulation Summary ---")
    final_summary = results_data.get("final_summary", {})
    sim_events = results_data.get("simulation_events", {})
    snn_appearance_data = results_data.get("snn_appearance_data", [])
    sim_end_time = final_summary.get("simulation_end_time", "N/A")
    total_attempted = results_data.get("total_vnrs_processed_by_agent", 0) 
    total_successes = final_summary.get("total_vnrs_embedded_successfully", 0)
    print(f"Simulation ended at timestep: {sim_end_time}")
    print(f"Total VNRs processed by agent: {total_attempted}")
    print(f"Total successful VNR embeddings: {total_successes}")
    final_acc_rate = total_successes / total_attempted if total_attempted > 0 else 0
    print(f"Final acceptance rate: {final_acc_rate:.2%}")
    
    broken_vnrs_count = sim_events.get("total_vnrs_broken_by_dynamics", 0) 
    successful_reembeds_vnrs = sim_events.get("total_vnrs_node_re_embedded_successfully", 0) 
    
    if RE_EMBED_BROKEN_VNRS: 
        print(f"VNRs broken by dynamics: {broken_vnrs_count}")
        print(f"VNRs successfully re-embedded (nodes): {successful_reembeds_vnrs}") 
        re_embedding_acceptance_rate = successful_reembeds_vnrs / broken_vnrs_count if broken_vnrs_count > 0 else 0
        print(f"Re-embedding Acceptance Rate for Broken VNRs: {re_embedding_acceptance_rate:.2%}")
    else: print("Re-embedding was disabled for this simulation run.")
    
    total_air_snn_added = 0; total_leo_snn_added = 0
    if snn_appearance_data: 
        last_appearance_entry = snn_appearance_data[-1] 
        total_air_snn_added = last_appearance_entry.get('air', 0)
        total_leo_snn_added = last_appearance_entry.get('leo', 0)
    print(f"Total AIR SNNs added (dynamically): {total_air_snn_added}")
    print(f"Total LEO SNNs added (dynamically): {total_leo_snn_added}")
    print("-----------------------------------")

if __name__ == '__main__':
    results_file_arg = None
    if len(os.sys.argv) > 1:
        results_file_arg = os.sys.argv[1]
        if not os.path.exists(results_file_arg): print(f"Error: Provided results file not found: {results_file_arg}"); exit()
    else: 
        results_dir_path = "ppo_vne_simulation_results/"
        latest_file = None; latest_mod_time = 0 
        if os.path.exists(results_dir_path):
            for f_name in os.listdir(results_dir_path):
                if f_name.startswith("sim_results_") and f_name.endswith(".json"):
                    f_path = os.path.join(results_dir_path, f_name)
                    try: 
                        f_mod_time = os.path.getmtime(f_path)
                        if f_mod_time > latest_mod_time: latest_mod_time = f_mod_time; latest_file = f_path
                    except OSError: print(f"Warning: Could not access metadata for file {f_path}"); continue 
        results_file_arg = latest_file
    if results_file_arg and os.path.exists(results_file_arg):
        print(f"Using results file: {results_file_arg}"); main(results_file_arg)
    else: print("No results file specified or found.")
