#!/usr/bin/env python3
"""
Plot gray failure sweep results: 3 CDF plots.

1. CDF of relative error magnitudes (flowSim and M4 vs UNISON ground truth)
2. CDF of signed relative errors (flowSim and M4 vs UNISON ground truth)
3. CDF of simulator runtimes (UNISON, flowSim, M4)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Color scheme matching util/plot.py
COLOR_LIST = ["crimson", "orange", "cornflowerblue", "blueviolet", "seagreen", "mediumpurple"]
LINESTYLE_LIST = ["-", "--", "--", "-.", ":"]
figure_size = (5, 3)
font_size = 18
ours = "m4"

def parse_endtoend_csv(filepath):
    """Parse EndToEnd.csv to extract total completion time in microseconds."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'total time,(\d+\.?\d*)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"⚠️  Error parsing {filepath}: {e}")
    return None


def parse_runtime_file(filepath):
    """Parse runtime.txt to extract execution duration in seconds."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('duration_seconds:'):
                    return float(line.split(':')[1].strip())
    except Exception as e:
        print(f"⚠️  Error parsing {filepath}: {e}")
    return None


def collect_results(results_dir):
    """
    Collect results from all simulator output directories.
    
    Returns:
        tuple: (completion_times, runtimes)
            completion_times: {simulator: {(n, r): time_us}}
            runtimes: {simulator: {(n, r): duration_sec}}
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return {}, {}
    
    completion_times = {'ns3': {}, 'flowsim': {}, 'm4': {}}
    runtimes = {'ns3': {}, 'flowsim': {}, 'm4': {}}
    
    pattern = re.compile(r'n_(\d+)_r_(\d+)_(ns3|flowsim|m4)')
    
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        match = pattern.match(subdir.name)
        if not match:
            continue
        
        n = int(match.group(1))
        r = int(match.group(2))
        simulator = match.group(3)
        
        # Parse EndToEnd.csv
        csv_file = subdir / 'EndToEnd.csv'
        if csv_file.exists():
            time_us = parse_endtoend_csv(csv_file)
            if time_us is not None:
                completion_times[simulator][(n, r)] = time_us
        
        # Parse runtime.txt
        runtime_file = subdir / 'runtime.txt'
        if runtime_file.exists():
            duration_sec = parse_runtime_file(runtime_file)
            if duration_sec is not None:
                runtimes[simulator][(n, r)] = duration_sec
    
    return completion_times, runtimes


def plot_signed_error_cdf(completion_times, output_file='gray_failure_signed_errors.png'):
    """Plot CDF of signed relative errors (flowSim and M4 vs UNISON)."""
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    # orange (flowSim), cornflowerblue (M4)
    colors = {'flowsim': COLOR_LIST[1], 'm4': COLOR_LIST[2]}
    labels = {'flowsim': 'flowSim', 'm4': ours}
    
    ns3_configs = set(completion_times['ns3'].keys())
    
    for idx, simulator in enumerate(['flowsim', 'm4']):
        sim_configs = set(completion_times[simulator].keys())
        common_configs = sorted(ns3_configs & sim_configs)
        
        if not common_configs:
            print(f"⚠️  No common configs for {simulator}")
            continue
        
        errors = []
        for config in common_configs:
            ns3_time = completion_times['ns3'][config]
            sim_time = completion_times[simulator][config]
            error = (sim_time - ns3_time) / ns3_time * 100  # Signed relative error in %
            errors.append(error)
        
        errors = np.array(errors)
        errors = errors[~np.isnan(errors)]
        
        # For this plot, use signed errors (not magnitude)
        data = errors
        data_size = len(data)
        
        # Use histogram method for CDF
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        counts = counts.astype(float) / data_size
        cdf = np.cumsum(counts)
        cdf = 100 * cdf / cdf[-1]
        
        ax.plot(bin_edges[0:-1], cdf,
                label=labels[simulator],
                color=colors[simulator],
                linestyle=LINESTYLE_LIST[idx],
                linewidth=2)
        
        # Print statistics with signed errors
        print(f"\n📊 {simulator.upper()} Signed Error Statistics:")
        print(f"   Mean: {np.mean(errors):+.2f}%")
        print(f"   Median: {np.median(errors):+.2f}%")
        print(f"   MAE: {np.mean(np.abs(errors)):.2f}%")
        print(f"   Range: [{np.min(errors):+.2f}%, {np.max(errors):+.2f}%]")
    
    plt.xlabel('Signed relative error (%)', fontsize=font_size)
    plt.ylabel('CDF (%)', fontsize=font_size)
    plt.ylim((0, 100))
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    
    legend_properties = {"size": font_size}
    plt.legend(prop=legend_properties, frameon=False, loc=4)
    
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_error_cdf(completion_times, output_file='gray_failure_errors.png'):
    """Plot 2: CDF of relative errors (flowSim and m4 vs UNISON)."""
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    # orange (flowSim), cornflowerblue (M4)
    colors = {'flowsim': COLOR_LIST[1], 'm4': COLOR_LIST[2]}
    labels = {'flowsim': 'flowSim', 'm4': ours}
    
    ns3_configs = set(completion_times['ns3'].keys())
    
    for idx, simulator in enumerate(['flowsim', 'm4']):
        sim_configs = set(completion_times[simulator].keys())
        common_configs = sorted(ns3_configs & sim_configs)
        
        if not common_configs:
            print(f"⚠️  No common configs for {simulator}")
            continue
        
        errors = []
        for config in common_configs:
            ns3_time = completion_times['ns3'][config]
            sim_time = completion_times[simulator][config]
            error = (sim_time - ns3_time) / ns3_time * 100  # Signed relative error in %
            errors.append(error)
        
        errors = np.array(errors)
        errors = errors[~np.isnan(errors)]
        
        # For CDF plot, use absolute values (magnitude of error)
        data = np.abs(errors)
        data_size = len(data)
        
        # Use histogram method for CDF
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        counts = counts.astype(float) / data_size
        cdf = np.cumsum(counts)
        cdf = 100 * cdf / cdf[-1]
        
        ax.plot(bin_edges[0:-1], cdf,
                label=labels[simulator],
                color=colors[simulator],
                linestyle=LINESTYLE_LIST[idx],
                linewidth=2)
        
        # Print statistics with signed errors
        print(f"\n📊 {simulator.upper()} Error Statistics:")
        print(f"   Mean: {np.mean(errors):+.2f}%")
        print(f"   Median: {np.median(errors):+.2f}%")
        print(f"   MAE: {np.mean(np.abs(errors)):.2f}%")
        print(f"   Range: [{np.min(errors):+.2f}%, {np.max(errors):+.2f}%]")
    
    plt.xlabel('Magnitude of relative error (%)', fontsize=font_size)
    plt.ylabel('CDF (%)', fontsize=font_size)
    plt.ylim((0, 100))
    plt.xlim(left=0.01)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    
    legend_properties = {"size": font_size}
    plt.legend(prop=legend_properties, frameon=False, loc=4)
    
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_runtime_cdf(runtimes, output_file='gray_failure_runtimes.png'):
    """Plot 3: CDF of simulator execution runtimes."""
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.set_xscale("log")
    
    # crimson (UNISON), orange (flowSim), cornflowerblue (M4)
    colors = {'ns3': COLOR_LIST[0], 'flowsim': COLOR_LIST[1], 'm4': COLOR_LIST[2]}
    labels = {'ns3': 'UNISON', 'flowsim': 'flowSim', 'm4': ours}
    
    for idx, simulator in enumerate(['ns3', 'flowsim', 'm4']):
        times = list(runtimes[simulator].values())
        if not times:
            print(f"⚠️  No runtime data for {simulator}")
            continue
        
        data = np.array(times)
        data = data[~np.isnan(data)]
        data_size = len(data)
        
        # Use histogram method for CDF
        data_set = sorted(set(data))
        bins = np.append(data_set, data_set[-1] + 1)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        counts = counts.astype(float) / data_size
        cdf = np.cumsum(counts)
        cdf = 100 * cdf / cdf[-1]
        
        ax.plot(bin_edges[0:-1], cdf,
                label=labels[simulator],
                color=colors[simulator],
                linestyle=LINESTYLE_LIST[idx],
                linewidth=2)
        
        print(f"\n⏱️  {simulator.upper()} Runtime: {len(times)} samples, "
              f"median {np.median(data):.1f}s, "
              f"range [{data.min():.1f}, {data.max():.1f}] s")
    
    # Compute speedups
    ns3_configs = set(runtimes['ns3'].keys())
    for simulator in ['flowsim', 'm4']:
        sim_configs = set(runtimes[simulator].keys())
        common_configs = ns3_configs & sim_configs
        if common_configs:
            speedups = [runtimes['ns3'][c] / runtimes[simulator][c] for c in common_configs]
            print(f"   {simulator.upper()} speedup vs UNISON: "
                  f"{np.median(speedups):.1f}x median, "
                  f"[{np.min(speedups):.1f}x, {np.max(speedups):.1f}x] range")
    
    plt.xlabel('Simulator execution time (s)', fontsize=font_size)
    plt.ylabel('CDF (%)', fontsize=font_size)
    plt.ylim((0, 100))
    plt.xlim(left=10)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    
    legend_properties = {"size": font_size}
    plt.legend(prop=legend_properties, frameon=False, loc=1)
    
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_mae_by_n(completion_times, output_file='gray_failure_mae_by_n.png'):
    """Plot MAE analysis by N: magnitude of relative error vs number of degraded GPUs."""
    # Compute MAE for each N
    ns3_configs = set(completion_times['ns3'].keys())
    
    n_values = list(range(2, 17))
    flowsim_mae_by_n = []
    m4_mae_by_n = []
    
    for n in n_values:
        fs_errors = []
        m4_errors = []
        for r in range(4, 11):
            if (n, r) in completion_times['ns3']:
                ns3_time = completion_times['ns3'][(n, r)]
                if (n, r) in completion_times['flowsim']:
                    fs_time = completion_times['flowsim'][(n, r)]
                    fs_errors.append(abs((fs_time - ns3_time) / ns3_time * 100))
                if (n, r) in completion_times['m4']:
                    m4_time = completion_times['m4'][(n, r)]
                    m4_errors.append(abs((m4_time - ns3_time) / ns3_time * 100))
        
        flowsim_mae_by_n.append(np.mean(fs_errors) if fs_errors else np.nan)
        m4_mae_by_n.append(np.mean(m4_errors) if m4_errors else np.nan)
    
    # Create figure with matching scatter plot style
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    ax.plot(n_values, flowsim_mae_by_n, '^-', color=COLOR_LIST[1], linewidth=2, 
            markersize=10, label='flowSim', alpha=0.8, markeredgewidth=1.5)
    ax.plot(n_values, m4_mae_by_n, 'x-', color=COLOR_LIST[2], linewidth=2, 
            markersize=12, label=ours, alpha=0.8, markeredgewidth=2)
    ax.set_xlabel('Number of Degraded GPUs (N)', fontsize=font_size)
    ax.set_ylabel('Mean magnitude of\nrelative error (%)', fontsize=font_size-2)
    ax.set_ylim([0, 35])
    ax.margins(y=0.1)
    ax.tick_params(axis='both', labelsize=font_size)
    
    legend_properties = {"size": font_size}
    ax.legend(prop=legend_properties, frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_mae_by_r(completion_times, output_file='gray_failure_mae_by_r.png'):
    """Plot MAE analysis by R: magnitude of relative error vs reduction factor."""
    # Compute MAE for each R
    ns3_configs = set(completion_times['ns3'].keys())
    
    r_values = list(range(4, 11))
    flowsim_mae_by_r = []
    m4_mae_by_r = []
    
    for r in r_values:
        fs_errors = []
        m4_errors = []
        for n in range(2, 17):
            if (n, r) in completion_times['ns3']:
                ns3_time = completion_times['ns3'][(n, r)]
                if (n, r) in completion_times['flowsim']:
                    fs_time = completion_times['flowsim'][(n, r)]
                    fs_errors.append(abs((fs_time - ns3_time) / ns3_time * 100))
                if (n, r) in completion_times['m4']:
                    m4_time = completion_times['m4'][(n, r)]
                    m4_errors.append(abs((m4_time - ns3_time) / ns3_time * 100))
        
        flowsim_mae_by_r.append(np.mean(fs_errors) if fs_errors else np.nan)
        m4_mae_by_r.append(np.mean(m4_errors) if m4_errors else np.nan)
    
    # Create figure with matching scatter plot style
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    ax.plot(r_values, flowsim_mae_by_r, '^-', color=COLOR_LIST[1], linewidth=2, 
            markersize=10, label='flowSim', alpha=0.8, markeredgewidth=1.5)
    ax.plot(r_values, m4_mae_by_r, 'x-', color=COLOR_LIST[2], linewidth=2, 
            markersize=12, label=ours, alpha=0.8, markeredgewidth=2)
    ax.set_xlabel('Reduction Factor (R)', fontsize=font_size)
    ax.set_ylabel('Mean magnitude of\nrelative error (%)', fontsize=font_size-2)
    ax.set_ylim([0, 35])
    ax.margins(y=0.1)
    ax.tick_params(axis='both', labelsize=font_size)
    
    legend_properties = {"size": font_size}
    ax.legend(prop=legend_properties, frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_scatter_by_n(completion_times, output_file='gray_failure_scatter_n8.png'):
    """Plot scatter plot for N=8: show completion times vs R for all simulators."""
    n = 8  # Only plot N=8
    
    # Match reference figure style
    colors = {'ns3': COLOR_LIST[0], 'flowsim': COLOR_LIST[1], 'm4': COLOR_LIST[2]}
    labels = {'ns3': 'UNISON', 'flowsim': 'flowSim', 'm4': ours}
    markers = {'ns3': 'o', 'flowsim': '^', 'm4': 'x'}
    markersizes = {'ns3': 10, 'flowsim': 10, 'm4': 12}
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    
    has_data = False
    for simulator in ['ns3', 'flowsim', 'm4']:
        # Collect data for N=8
        r_values = []
        times = []
        for (config_n, config_r), time_us in completion_times[simulator].items():
            if config_n == n:
                r_values.append(config_r)
                times.append(time_us / 1000.0)  # Convert to ms
        
        if r_values:
            has_data = True
            # Sort by R
            sorted_pairs = sorted(zip(r_values, times))
            r_sorted, times_sorted = zip(*sorted_pairs)
            
            # Plot with markers and connecting lines
            ax.plot(r_sorted, times_sorted,
                   label=labels[simulator],
                   color=colors[simulator],
                   marker=markers[simulator],
                   markersize=markersizes[simulator],
                   linewidth=2,
                   linestyle='-',
                   markeredgewidth=2 if simulator == 'm4' else 1.5,
                   alpha=0.8)
    
    if not has_data:
        print("⚠️  No data for N=8 scatter plot")
        plt.close()
        return
    
    # Set y-axis limits with some padding to prevent cutoff
    ax.margins(y=0.1)  # Add 10% margin on y-axis
    
    plt.xlabel('Reduction Factor (R)', fontsize=font_size)
    plt.ylabel('Application\nCompletion Time (ms)', fontsize=font_size-4)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    
    legend_properties = {"size": font_size}
    plt.legend(prop=legend_properties, frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot gray failure sweep results: CDFs and scatter plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./plot_gray_failure_results.py
  ./plot_gray_failure_results.py --results-dir ./my_results
        """
    )
    
    parser.add_argument('--results-dir', default='./results_gray_failures',
                        help='Directory containing simulation results')
    parser.add_argument('--error-output', default='gray_failure_errors.png',
                        help='Output file for error magnitude CDF')
    parser.add_argument('--signed-error-output', default='gray_failure_signed_errors.png',
                        help='Output file for signed error CDF')
    parser.add_argument('--runtime-output', default='gray_failure_runtimes.png',
                        help='Output file for runtime CDF')
    parser.add_argument('--scatter-output', default='gray_failure_scatter_n8.png',
                        help='Output file for N=8 scatter plot')
    parser.add_argument('--mae-n-output', default='gray_failure_mae_by_n.png',
                        help='Output file for MAE by N plot')
    parser.add_argument('--mae-r-output', default='gray_failure_mae_by_r.png',
                        help='Output file for MAE by R plot')
    
    args = parser.parse_args()
    
    print("🚀 Collecting results...")
    completion_times, runtimes = collect_results(args.results_dir)
    
    # Check data
    n_completion = sum(len(v) for v in completion_times.values())
    n_runtime = sum(len(v) for v in runtimes.values())
    
    if n_completion == 0:
        print("❌ No completion time data found!")
        return
    
    print(f"\n📊 Found {n_completion} completion time results")
    print(f"⏱️  Found {n_runtime} runtime results")
    
    # Generate plots
    print("\n📈 Generating error CDF plots...")
    plot_error_cdf(completion_times, args.error_output)
    plot_signed_error_cdf(completion_times, args.signed_error_output)
    
    if n_runtime > 0:
        print("\n⏱️  Generating runtime CDF plot...")
        plot_runtime_cdf(runtimes, args.runtime_output)
    else:
        print("⚠️  No runtime data, skipping runtime plot")
    
    print("\n📊 Generating scatter plot for N=8...")
    plot_scatter_by_n(completion_times, args.scatter_output)
    
    print("\n📊 Generating MAE analysis plots...")
    plot_mae_by_n(completion_times, args.mae_n_output)
    plot_mae_by_r(completion_times, args.mae_r_output)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
