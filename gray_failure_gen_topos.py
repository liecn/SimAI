#!/usr/bin/env python3
"""
Generate structured gray failure topologies for SimAI experiments.
Matches the pattern from topo_N{4,8,16}_M*.txt structured topologies.
"""

import random
import os


def get_structured_degraded_gpus(num_degraded_gpus, seed=None):
    """
    Generate FIXED server-aligned degraded GPU patterns.
    Matches the exact patterns from topo_N{8,16}_M*.txt for consistency.
    
    Pattern strategy (FIXED for all R values):
    - N=8: GPUs [0-3, 8-11] (Server 0 + Server 1, first half of each)
    - N=16: GPUs [0-7, 16-23] (Full Server 0 + Full Server 2)
    - Other N: Follow similar server-aligned, contiguous block patterns
    """
    degraded_gpus = set()
    
    if num_degraded_gpus <= 8:
        # N=8 or less: Split evenly between Server 0 and Server 1
        # Pattern: [0-3] + [8-11] for N=8 (matches topo_N8_M*.txt)
        half = num_degraded_gpus // 2
        remainder = num_degraded_gpus % 2
        
        # First half in Server 0
        for i in range(half + remainder):
            degraded_gpus.add(i)
        
        # Second half in Server 1
        for i in range(half):
            degraded_gpus.add(8 + i)
    
    elif num_degraded_gpus <= 16:
        # N=9-16: Use Server 0 and Server 2 (skip Server 1)
        # Pattern for N=16: [0-7, 16-23] (matches topo_N16_M*.txt)
        if num_degraded_gpus == 16:
            # Full Server 0 and full Server 2
            for i in range(8):
                degraded_gpus.add(i)      # Server 0
                degraded_gpus.add(16 + i) # Server 2
        else:
            # N=9-15: Fill Server 0 first, then Server 2
            for i in range(min(8, num_degraded_gpus)):
                degraded_gpus.add(i)
            
            remaining = num_degraded_gpus - 8
            if remaining > 0:
                for i in range(remaining):
                    degraded_gpus.add(16 + i)
    
    else:
        # N=17-23: Continue with servers in order (0, 1, 2, 3)
        full_servers = num_degraded_gpus // 8
        remainder = num_degraded_gpus % 8
        
        # Degrade full servers (0, 1, 2)
        for server_idx in range(full_servers):
            for i in range(8):
                degraded_gpus.add(server_idx * 8 + i)
        
        # Add remainder to next server
        if remainder > 0:
            for i in range(remainder):
                degraded_gpus.add(full_servers * 8 + i)
    
    return degraded_gpus


def generate_topology(num_degraded_gpus, reduction_factor, seed=None):
    """Generate topology file with structured degraded GPUs."""
    degraded_gpus = get_structured_degraded_gpus(num_degraded_gpus, seed=seed)
    
    normal_nvswitch_bw = 2400
    normal_rail_bw = 400
    degraded_nvswitch_bw = normal_nvswitch_bw // reduction_factor
    degraded_rail_bw = normal_rail_bw // reduction_factor
    
    lines = []
    lines.append("45 8 4 9 72 A100")
    lines.append("32 33 34 35 36 37 38 39 40 41 42 43 44")
    
    for gpu in range(32):
        nvswitch = 32 + (gpu // 8)
        half_server = (gpu % 8) // 4
        rail_offset = gpu % 4
        rail_switch = 36 + (half_server * 4) + rail_offset
        
        if gpu in degraded_gpus:
            nvswitch_bw = degraded_nvswitch_bw
            rail_bw = degraded_rail_bw
        else:
            nvswitch_bw = normal_nvswitch_bw
            rail_bw = normal_rail_bw
        
        lines.append(f"{gpu} {nvswitch} {nvswitch_bw}Gbps 0.000025ms 0")
        lines.append(f"{gpu} {rail_switch} {rail_bw}Gbps 0.0005ms 0")
    
    for rail_switch in range(36, 44):
        lines.append(f"{rail_switch} 44 400Gbps 0.0005ms 0")
    
    lines.append("")
    return lines, degraded_gpus


def main():
    output_dir = "./example/gray_failures"
    os.makedirs(output_dir, exist_ok=True)
    
    num_gpus_range = range(2, 17)
    reduction_factors = range(4, 11)
    
    metadata_lines = ["# Gray Failure Topology Metadata",
                     "# Format: filename, num_degraded_gpus, reduction_factor, degraded_gpu_ids", ""]
    
    topo_count = 0
    for num_degraded in num_gpus_range:
        for reduction_factor in reduction_factors:
            seed = num_degraded * 1000 + reduction_factor
            lines, degraded_gpus = generate_topology(num_degraded, reduction_factor, seed=seed)
            
            filename = f"gray_topo_N{num_degraded}_R{reduction_factor}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))
            
            degraded_list = ','.join(map(str, sorted(degraded_gpus)))
            metadata_lines.append(f"{filename}, {num_degraded}, 1/{reduction_factor}, [{degraded_list}]")
            topo_count += 1
            
            if topo_count % 20 == 0:
                print(f"Generated {topo_count} topologies...")
    
    metadata_path = os.path.join(output_dir, "topology_metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))
    
    print(f"\n✅ Generated {topo_count} structured gray failure topologies")
    print(f"   Output: {output_dir}/")
    print(f"   N=[{min(num_gpus_range)},{max(num_gpus_range)+1}), R=[1/{min(reduction_factors)},1/{max(reduction_factors)})")
    
    print(f"\n📝 Example patterns:")
    for n in [8, 12, 16]:
        seed = n * 1000 + 4
        gpus = get_structured_degraded_gpus(n, seed=seed)
        print(f"   N={n}: {sorted(gpus)}")


if __name__ == "__main__":
    main()

