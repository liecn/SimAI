#!/usr/bin/env python3
import argparse, re
from collections import defaultdict
import matplotlib.pyplot as plt

font_size = 20
def parse_topo(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = lines[:2]
    edges = []
    for ln in lines[2:]:
        parts = ln.split()
        if len(parts) < 4: 
            continue
        try:
            u = int(parts[0]); v = int(parts[1])
            bw = int(re.match(r"(\d+)", parts[2]).group(1))
            lat = float(re.match(r"([\d\.]+)", parts[3]).group(1))
        except Exception:
            continue
        edges.append((u,v,bw,lat))
    return header, edges

def build_graph(edges):
    adj = defaultdict(dict)
    nodes=set()
    for u,v,bw,_ in edges:
        adj[u][v]=bw; adj[v][u]=bw
        nodes.add(u); nodes.add(v)
    spine = max(nodes)
    L2 = [n for n in nodes if n>=32 and n!=spine and spine in adj[n]]
    L1 = [n for n in nodes if n>=32 and n!=spine and n not in L2]
    l2_to_gpus = {l2: sorted([(g, adj[l2][g]) for g in adj[l2] if g<32]) for l2 in sorted(L2)}
    l2_uplink = {l2: adj[l2][spine] for l2 in L2}
    gpu_to_l2 = {g:(l2, adj[l2][g]) for l2 in L2 for g,_ in l2_to_gpus[l2]}
    return adj, spine, sorted(L1), sorted(L2), l2_to_gpus, l2_uplink, gpu_to_l2

def plot_groups_layers(out_path, header, adj, spine, L1, L2):
    nodes = set(adj.keys())
    gpus = sorted([n for n in nodes if n < 32])
    pos = {}
    
    # Position GPUs in a single row
    gpu_spacing = 1.0
    for i,g in enumerate(gpus):
        pos[g] = (i * gpu_spacing, 1)
    
    # Position NVSwitches (L1) below GPUs
    nvswitch_width = (len(gpus) - 1) * gpu_spacing
    for i,s in enumerate(L1):
        x_pos = (i + 0.5) * (nvswitch_width / len(L1))
        pos[s] = (x_pos, 0)
    
    # Position Rail switches (L2) above GPUs
    for i,s in enumerate(L2):
        x_pos = i * (nvswitch_width / (len(L2) - 1)) if len(L2) > 1 else nvswitch_width / 2
        pos[s] = (x_pos, 2)
    
    # Position Spine at top center
    pos[spine] = (nvswitch_width / 2, 3)

    plt.figure(figsize=(20,8))
    
    # Draw links with different styles based on bandwidth
    for u in adj:
        for v,bw in adj[u].items():
            if u<v:
                x1,y1=pos[u]; x2,y2=pos[v]
                
                # Categorize by bandwidth:
                # - 2400 Gbps: GPU ↔ NVSwitch (NVLink, intra-node)
                # - 400 Gbps: Rail ↔ Spine (network backbone)
                # - 100 Gbps: GPU ↔ Rail (NIC, inter-node)
                
                if bw >= 2400:
                    # GPU-NVSwitch (NVLink): thick solid black line
                    ls = '-'
                    lw = 3.5
                    alpha = 0.6
                    color = 'black'
                    label_bw = '2400G'
                elif bw >= 400:
                    # Rail-Spine (network backbone): dashed blue line
                    ls = '--'
                    lw = 2.5
                    alpha = 0.5
                    color = 'blue'
                    label_bw = '400G'
                elif bw >= 100:
                    # GPU-Rail (NIC): dotted red line (thinner)
                    ls = ':'
                    lw = 1.5
                    alpha = 0.4
                    color = 'red'
                    label_bw = '100G'
                else:
                    # Degraded/unknown: thin dotted orange
                    ls = ':'
                    lw = 1.0
                    alpha = 0.3
                    color = 'orange'
                    label_bw = f'{bw}G'
                
                plt.plot([x1,x2],[y1,y2], ls, linewidth=lw, color=color, alpha=alpha, zorder=0)
    groupA = [g for g in gpus if 0<=g<=15]
    groupB = [g for g in gpus if 16<=g<=31]
    
    # Create custom legend entries for link types (bandwidth hierarchy)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=3.5, linestyle='-', alpha=0.6, 
               label='GPU↔NVSwitch: 2400 Gbps (NVLink)'),
        Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', alpha=0.5, 
               label='Rail↔Spine: 400 Gbps (Network Backbone)'),
        Line2D([0], [0], color='red', linewidth=1.5, linestyle=':', alpha=0.4, 
               label='GPU↔Rail: 100 Gbps (NIC, Multi-rail)'),
    ]
    
    plt.scatter([pos[g][0] for g in groupA],[pos[g][1] for g in groupA], marker='s', s=400, label="Group A (0–15)")
    plt.scatter([pos[g][0] for g in groupB],[pos[g][1] for g in groupB], marker='^', s=400, label="Group B (16–31)")
    plt.scatter([pos[s][0] for s in L1],[pos[s][1] for s in L1], marker='o', s=400, label="NVSwitch")
    plt.scatter([pos[s][0] for s in L2],[pos[s][1] for s in L2], marker='D', s=400, label="Rail switch")
    plt.scatter([pos[spine][0]],[pos[spine][1]], marker='^', s=400, label="Spine")
    # GPU labels
    for g in gpus:
        x,y = pos[g]
        plt.text(x, y+0.2, str(g), ha='center', va='top', fontsize=font_size-4)
    
    # Rail switch labels (L2)
    for s in L2:
        x,y = pos[s]
        plt.text(x, y+0.15, str(s), ha='center', va='bottom', fontsize=font_size-2, color='darkgreen')
    
    # Spine label
    x,y = pos[spine]
    plt.text(x, y+0.15, str(spine), ha='center', va='bottom', fontsize=font_size, color='darkred', weight='bold')
    
    # NVSwitch labels (L1)
    for s in L1:
        x,y = pos[s]
        plt.text(x+0.8, y+0.05, str(s), ha='center', va='top', fontsize=font_size-2, color='darkgreen')
    plt.xticks([])
    plt.yticks([0,1,2,3], ["NVSwitch","GPUs","Rail switch","Spine"], fontsize=font_size+5)
    # plt.title(f"Topology layers with groups (IDs shown)\n{' '.join(header)}")
    plt.xlabel("Logical position (rank order)", fontsize=font_size+5)
    plt.grid(True, linestyle=":")
    
    # Add legend with both link types and node types
    plt.legend(handles=legend_elements, loc="upper left", fontsize=font_size-2, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser(description="SimAI topology visualizations - groups layer view")
    ap.add_argument("--topo", default="example/topo.txt", help="path to topology file (default: example/topo.txt)")
    ap.add_argument("--out", default="simai_topo_groups.png", help="output filename (default: simai_topo_groups.png)")
    args = ap.parse_args()
    
    header, edges = parse_topo(args.topo)
    adj, spine, L1, L2, l2_to_gpus, l2_uplink, gpu_to_l2 = build_graph(edges)
    plot_groups_layers(args.out, header, adj, spine, L1, L2)
    print(f"✅ Saved topology visualization to: {args.out}")

if __name__ == "__main__":
    main()
