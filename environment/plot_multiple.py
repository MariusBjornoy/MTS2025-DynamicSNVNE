import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
import csv

# --- Configuration ---
RESULTS_DIR = "Final_ttl/"
PLOT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "summary_plots_all")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- Font Size Configuration for Reports (Even Bigger) ---
TITLE_FONTSIZE = 40
SUBTITLE_FONTSIZE = 26
AXIS_LABEL_FONTSIZE = 30
TICK_LABEL_FONTSIZE = 20
PIE_CHART_TEXT_FONTSIZE = 28
BAR_LABEL_FONTSIZE = 26

# Global variable for RE_EMBED_BROKEN_VNRS
RE_EMBED_BROKEN_VNRS = False 

def load_all_results(results_dir):
    """ Loads all JSON result files from the specified directory. """
    json_files = glob.glob(os.path.join(results_dir, "sim_results_*.json"))
    if not json_files:
        print(f"Error: No result files found in '{results_dir}'. Please run the parameter sweep simulation first.")
        return None
    
    all_results = []
    # Sort files based on arrival rate then TTL from filename for consistent plot layout
    def sort_key(filepath):
        try:
            parts = os.path.basename(filepath).split('_')
            arr_mean = float(parts[parts.index('arr')+1])
            ttl_mean = float(parts[parts.index('ttl')+1])
            return arr_mean, ttl_mean
        except (ValueError, IndexError):
            return 0, 0 # Fallback for unexpected filenames
            
    for f_path in sorted(json_files, key=sort_key):
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load or parse {f_path}. Error: {e}")
    return all_results

# --- PLOTTING FUNCTIONS ---

def plot_line_chart_grid(all_results, plot_type, title, ylabel, xlabel, is_percent=False):
    """ Creates a grid of subplots for a given time-series metric with enhanced labeling. """
    num_results = len(all_results)
    if num_results == 0: return
    cols = 3; rows = math.ceil(num_results / cols) if cols > 0 else 0
    fig, axes = plt.subplots(rows, cols, figsize=(25, 8 * rows), sharex=True, sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.98); axes = axes.flatten()

    for i, result_data in enumerate(all_results):
        ax = axes[i]
        config = result_data.get("configuration", {})
        arr_mean = config.get("vnr_inter_arrival_mean", 0)
        ttl_mean = config.get("vnr_mean_ttl", "N/A")
        
        # Format arrival rate as "X/100"
        arrivals_per_100_int = round(100 / arr_mean) if arr_mean > 0 else 0
        arrival_str = f"{arrivals_per_100_int}/100"
        ax.set_title(f"Arrival Rate: {arrival_str} ts, TTL Mean: {ttl_mean}", fontsize=SUBTITLE_FONTSIZE)
        
        data_to_plot = []; x_axis_data = []
        if plot_type == 'acceptance_rate_per_vnr':
            vnr_details = result_data.get("vnr_details", [])
            if vnr_details:
                successful_embeddings = 0
                for j, vnr in enumerate(vnr_details):
                    if vnr.get('status') == 'successfully_embedded': successful_embeddings += 1
                    data_to_plot.append(successful_embeddings / (j + 1))
                x_axis_data = range(1, len(vnr_details) + 1)
        elif plot_type == 'concurrency':
            concurrency_data = result_data.get("concurrency_data", []); x_axis_data = [dp['time'] for dp in concurrency_data]; data_to_plot = [dp['active_vnrs'] for dp in concurrency_data]
        elif plot_type == 'cumulative_broken_vnrs':
             broken_vnr_data = result_data.get("broken_vnr_progression_data", []); x_axis_data = [dp['time'] for dp in broken_vnr_data]; data_to_plot = [dp['cumulative_broken_vnrs'] for dp in broken_vnr_data]
        elif plot_type == 'rc_ratio_per_vnr':
             vnr_details = result_data.get("vnr_details", [])
             if vnr_details:
                for j, vnr in enumerate(vnr_details):
                    if vnr.get('status') == 'successfully_embedded':
                        revenue = vnr.get('revenue', 0.0); cost = vnr.get('cost', 0.0)
                        ratio = revenue / cost if cost > 0 else 5.0 if revenue > 0 else 1.0
                        x_axis_data.append(j + 1); data_to_plot.append(ratio)
        
        if data_to_plot: 
            if plot_type == 'rc_ratio_per_vnr': ax.scatter(x_axis_data, data_to_plot, marker='o', s=10, alpha=0.6, color='purple')
            else: ax.plot(x_axis_data, data_to_plot, linestyle='-', marker=None, color='teal')
        else: ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.grid(True, linestyle='--', alpha=0.6)
        if is_percent: ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        
        if i >= (rows - 1) * cols: ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
        if i % cols == 0: ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    for j in range(num_results, len(axes)): axes[j].axis('off')
    # Adjusted rect for tight_layout to prevent labels from being cut off
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.95]); plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"summary_{plot_type}.png")
    plt.savefig(plot_filename); plt.close(fig); print(f"Summary plot saved: {plot_filename}")

def plot_histogram_grid(all_results, data_source, data_key, title, xlabel, bins=30):
    """ Generates a grid of histograms with enhanced labeling. """
    num_results = len(all_results)
    if num_results == 0: return
    cols = 3; rows = math.ceil(num_results / cols) if cols > 0 else 0
    fig, axes = plt.subplots(rows, cols, figsize=(25, 8 * rows), sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.98); axes = axes.flatten()
    for i, result_data in enumerate(all_results):
        ax = axes[i]
        config = result_data.get("configuration", {})
        arr_mean = config.get("vnr_inter_arrival_mean", 0)
        ttl_mean = config.get("vnr_mean_ttl", "N/A")
        
        # Format arrival rate as "X/100"
        arrivals_per_100_int = round(100 / arr_mean) if arr_mean > 0 else 0
        arrival_str = f"{arrivals_per_100_int}/100"
        ax.set_title(f"Arrival Rate: {arrival_str} ts, TTL Mean: {ttl_mean}", fontsize=SUBTITLE_FONTSIZE)
        
        values = []
        if data_key is None: values = result_data.get(data_source, [])
        else: 
            source_data = result_data.get(data_source, {})
            if data_source == "vnr_details":
                if "cost" in data_key or "revenue" in data_key:
                    values = [vnr[data_key] for vnr in source_data if vnr.get('status') == 'successfully_embedded' and data_key in vnr and vnr[data_key] is not None]
                else: values = [vnr[data_key] for vnr in source_data if data_key in vnr and vnr[data_key] is not None]
            else: values = source_data.get(data_key, [])
        
        if values: ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)
        else: ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i >= (rows-1) * cols : ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
        if i % cols == 0: ax.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    for j in range(num_results, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.95]); plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"summary_dist_{data_source.replace('/','')}_{data_key if data_key else 'data'}.png")
    plt.savefig(plot_filename); plt.close(fig); print(f"Summary plot saved: {plot_filename}")

def plot_pie_chart_grid(all_results, save_path):
    """ Generates a grid of pie charts with enhanced labeling. """
    num_results = len(all_results)
    if num_results == 0: return
    cols = 3; rows = math.ceil(num_results / cols) if cols > 0 else 0
    fig, axes = plt.subplots(rows, cols, figsize=(25, 9 * rows), squeeze=False)
    fig.suptitle('Re-embedding Acceptance Rate for Broken VNRs', fontsize=TITLE_FONTSIZE, y=0.98)
    axes = axes.flatten()
    for i, result_data in enumerate(all_results):
        ax = axes[i]
        config = result_data.get("configuration", {})
        arr_mean = config.get("vnr_inter_arrival_mean", 0)
        ttl_mean = config.get("vnr_mean_ttl", "N/A")
        
        # Format arrival rate as "X/100"
        arrivals_per_100_int = round(100 / arr_mean) if arr_mean > 0 else 0
        arrival_str = f"{arrivals_per_100_int}/100"
        ax.set_title(f"Arrival Rate: {arrival_str} ts, TTL Mean: {ttl_mean}", fontsize=SUBTITLE_FONTSIZE)
        
        events_summary = result_data.get("simulation_events", {}); re_embed_enabled = config.get("re_embed_broken_vnrs", False)
        if not re_embed_enabled: ax.text(0.5, 0.5, "Re-embedding Disabled", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE); continue
        total_broken = events_summary.get('total_vnrs_broken_by_dynamics', 0)
        total_success = events_summary.get('total_vnrs_node_re_embedded_successfully', 0)
        if total_broken == 0: ax.text(0.5, 0.5, "No VNRs Broken", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE); continue
        failed = total_broken - total_success; sizes = [max(0, total_success), max(0, failed)]
        labels = [f'Success ({total_success})', f'Fail ({failed})']; colors = ['lightgreen', 'lightcoral']
        explode = (0.1, 0) if total_success > 0 else (0, 0)
        if sum(sizes) > 0: ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': PIE_CHART_TEXT_FONTSIZE})
        else: ax.text(0.5, 0.5, "No Re-embedding Data", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE)
        ax.axis('equal')
    for j in range(num_results, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plot_filename = os.path.join(save_path, "summary_plot_rebed_acceptance.png")
    plt.savefig(plot_filename); plt.close(fig); print(f"Summary plot saved: {plot_filename}")

def plot_snn_churn_grid(all_results):
    """ Generates a grid of SNN churn plots with enhanced labeling. """
    num_results = len(all_results);
    if num_results == 0: return
    cols = 3; rows = math.ceil(num_results / cols) if cols > 0 else 0
    fig, axes = plt.subplots(rows, cols, figsize=(25, 8 * rows), sharex=True, sharey=True, squeeze=False)
    fig.suptitle('Cumulative SNN Churn (Added vs. Removed) Across Parameters', fontsize=TITLE_FONTSIZE, y=0.98); axes = axes.flatten()
    for i, result_data in enumerate(all_results):
        ax = axes[i]
        config = result_data.get("configuration", {})
        arr_mean = config.get("vnr_inter_arrival_mean", 0)
        ttl_mean = config.get("vnr_mean_ttl", "N/A")
        
        # Format arrival rate as "X/100"
        arrivals_per_100_int = round(100 / arr_mean) if arr_mean > 0 else 0
        arrival_str = f"{arrivals_per_100_int}/100"
        ax.set_title(f"Arrival Rate: {arrival_str} ts, TTL Mean: {ttl_mean}", fontsize=SUBTITLE_FONTSIZE)

        added_data = result_data.get("snn_appearance_data", [])
        removed_data = result_data.get("snn_removal_data", [])
        if added_data:
            times_added = [dp['time'] for dp in added_data]
            ax.plot(times_added, [dp['air'] for dp in added_data], label='Air Added', color='blue', linestyle='-')
            ax.plot(times_added, [dp['leo'] for dp in added_data], label='LEO Added', color='orange', linestyle='-')
        if removed_data:
            times_removed = [dp['time'] for dp in removed_data]
            ax.plot(times_removed, [-dp['air'] for dp in removed_data], label='Air Removed', color='blue', linestyle='--')
            ax.plot(times_removed, [-dp['leo'] for dp in removed_data], label='LEO Removed', color='orange', linestyle='--')
        ax.axhline(0, color='black', linewidth=0.8); ax.legend(loc='best', fontsize=TICK_LABEL_FONTSIZE); ax.grid(True, linestyle='--', alpha=0.6)
        
        if i >= (rows-1) * cols: ax.set_xlabel('Simulation Time steps [S]', fontsize=AXIS_LABEL_FONTSIZE)
        if i % cols == 0: ax.set_ylabel('Cumulative Node Count', fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    for j in range(num_results, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.95]); plot_filename = os.path.join(PLOT_OUTPUT_DIR, "summary_plot_snn_churn.png")
    plt.savefig(plot_filename); plt.close(fig); print(f"Summary plot saved: {plot_filename}")

def plot_events_summary_grid(all_results):
    """ Generates a grid of event summary bar charts with a fixed Y-axis and counts on bars. """
    num_results = len(all_results)
    if num_results == 0: return
    cols = 3; rows = math.ceil(num_results / cols) if cols > 0 else 0
    fig, axes = plt.subplots(rows, cols, figsize=(25, 8 * rows), sharey=True, squeeze=False)
    fig.suptitle('Summary of VNR Lifecycle Events Across Parameters', fontsize=TITLE_FONTSIZE, y=0.98); axes = axes.flatten()
    for i, result_data in enumerate(all_results):
        ax = axes[i]
        config = result_data.get("configuration", {})
        arr_mean = config.get("vnr_inter_arrival_mean", 0)
        ttl_mean = config.get("vnr_mean_ttl", "N/A")

        # Format arrival rate as "X/100"
        arrivals_per_100_int = round(100 / arr_mean) if arr_mean > 0 else 0
        arrival_str = f"{arrivals_per_100_int}/100"
        ax.set_title(f"Arrival Rate: {arrival_str} ts, TTL Mean: {ttl_mean}", fontsize=SUBTITLE_FONTSIZE)
        
        successful_embeddings = result_data.get("successful_embeddings", 0)
        failed_initial_embeddings = result_data.get("failed_initial_embeddings", 0)
        events_summary = result_data.get("simulation_events", {})
        labels = ['Initial Success', 'Initial Fail', 'Broken', 'Re-embedded']
        counts = [successful_embeddings, failed_initial_embeddings, events_summary.get('total_vnrs_broken_by_dynamics', 0), events_summary.get('total_vnrs_node_re_embedded_successfully', 0)]
        colors = ['green', 'red', 'orange', 'cyan']
        
        bars = ax.bar(labels, counts, color=colors)
        ax.set_ylabel("Count of VNRs", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='x', rotation=30, labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)

        # --- MODIFIED: Set a fixed Y-axis limit and add labels on top of bars ---
        ax.set_ylim(0, 1000)
            
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=BAR_LABEL_FONTSIZE)

    for j in range(num_results, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]); plot_filename = os.path.join(PLOT_OUTPUT_DIR, "summary_plot_lifecycle_events.png")
    plt.savefig(plot_filename); plt.close(fig); print(f"Summary plot saved: {plot_filename}")

def main():
    print(f"Loading results from directory: {RESULTS_DIR}")
    all_results_data = load_all_results(RESULTS_DIR)
    if not all_results_data: return

    global RE_EMBED_BROKEN_VNRS
    if all_results_data:
        RE_EMBED_BROKEN_VNRS = all_results_data[0].get("configuration", {}).get("re_embed_broken_vnrs", False)

    print("\n--- Generating Summary Plots ---")
    
    # Time-series performance plots
    plot_line_chart_grid(all_results_data, "acceptance_rate_per_vnr", "Acceptance Rate Progression", "Cumulative Acceptance Rate", "Number of VNRs Processed", is_percent=True)
    plot_line_chart_grid(all_results_data, "concurrency", "VNR Concurrency", "Number of Active VNRs", " Simularion Time steps [S]")
    plot_line_chart_grid(all_results_data, "cumulative_broken_vnrs", "Cumulative Broken VNRs", "Cumulative Broken VNRs", "Simulation Time steps [S]")
    plot_line_chart_grid(all_results_data, "rc_ratio_per_vnr", "Revenue to Cost Ratio per Successful VNR", "Individual R/C Ratio", "Number of VNRs Processed")

    # SNN Dynamics plots
    plot_snn_churn_grid(all_results_data)
    
    # Histogram distribution plots
    plot_histogram_grid(all_results_data, "initial_lifetimes", "air", "Distribution of Air SNN Lifetimes", "Lifetime (Timesteps) [S]")
    plot_histogram_grid(all_results_data, "initial_lifetimes", "leo", "Distribution of LEO SNN Lifetimes", "Lifetime (Timesteps) [S]")
    plot_histogram_grid(all_results_data, "inter_arrival_times", None, "Distribution of VNR Inter-Arrival Times", "Inter-Arrival Time (Timesteps)[S]", bins=50)
    plot_histogram_grid(all_results_data, "vnr_details", "cost", "Distribution of Cost per Successful VNR", "Cost")
    plot_histogram_grid(all_results_data, "vnr_details", "revenue", "Distribution of Revenue per Successful VNR", "Revenue")
    plot_histogram_grid(all_results_data, "vnr_details", "num_nodes", "Distribution of VNR Length (Nodes)", "Number of VNNs")
    plot_histogram_grid(all_results_data, "vnr_details", "ttl_actual_sampled", "Distribution of VNR TTLs (Actual Sampled)", "TTL (Timesteps)[S]")

    # Pie chart for re-embedding and bar chart for events
    plot_pie_chart_grid(all_results_data, PLOT_OUTPUT_DIR)
    plot_events_summary_grid(all_results_data)

    # --- Create Summary CSV ---
    summary_csv_path = os.path.join(PLOT_OUTPUT_DIR, "parameter_sweep_summary.csv")
    csv_data = []
    for result_data in all_results_data:
        config = result_data.get("configuration", {})
        final_summary = result_data.get("final_summary", {})
        sim_events = result_data.get("simulation_events", {})
        concurrency_data = result_data.get("concurrency_data", [])
        vnr_details = result_data.get("vnr_details", [])
        inter_arrival_times = result_data.get("inter_arrival_times", [])
        max_concurrency = max([dp['active_vnrs'] for dp in concurrency_data]) if concurrency_data else 0
        mean_vnr_lifetime_actual = np.mean([vnr['ttl_actual_sampled'] for vnr in vnr_details if 'ttl_actual_sampled' in vnr]) if vnr_details else 0
        mean_inter_arrival_actual = np.mean(inter_arrival_times) if inter_arrival_times else 0 
        
        # Format arrival rate for CSV
        arr_mean_config = config.get("vnr_inter_arrival_mean", 0)
        arrivals_per_100_int_config = round(100 / arr_mean_config) if arr_mean_config > 0 else 0
        arrival_str_config = f"{arrivals_per_100_int_config}/100"

        row_data = {
            'Arrival_Rate_Config': arrival_str_config,
            'TTL_Mean_Config': config.get("vnr_mean_ttl", "N/A"),
            'Final_Acceptance_Rate': final_summary.get("acceptance_rate", 0),
            'Max_Concurrent_VNRs': max_concurrency,
            'Total_Broken_VNRs': sim_events.get("total_vnrs_broken_by_dynamics", 0),
            'Arrival_Mean_Config': arr_mean_config,
            'Arrival_Mean_Actual': mean_inter_arrival_actual,
            'TTL_Mean_Actual': mean_vnr_lifetime_actual,
        }
        csv_data.append(row_data)
        
    if csv_data:
        try:
            with open(summary_csv_path, 'w', newline='') as csvfile:
                # Reordered fieldnames for clarity
                fieldnames = ['Arrival_Rate_Config', 'TTL_Mean_Config', 'Final_Acceptance_Rate', 'Max_Concurrent_VNRs', 'Total_Broken_VNRs', 'Arrival_Mean_Config', 'Arrival_Mean_Actual', 'TTL_Mean_Actual']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"Summary CSV file saved: {summary_csv_path}")
        except Exception as e: print(f"Error saving summary CSV: {e}")
    print("\n--- All plots and summary generation finished. ---")

if __name__ == '__main__':
    main()