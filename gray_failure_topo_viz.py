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
    for i,g in enumerate(gpus):
        pos[g] = (i, 1)
    for i,s in enumerate(L1):
        pos[s] = (i*(len(gpus)/max(1,len(L1))) + 3, 0)
    for i,s in enumerate(L2):
        pos[s] = (i*(len(gpus)/max(1,len(L2))) + 1, 2)
    pos[spine] = (len(gpus)/2, 3)

    plt.figure(figsize=(14,7))
    
    # Draw links with different styles based on bandwidth
    for u in adj:
        for v,bw in adj[u].items():
            if u<v:
                x1,y1=pos[u]; x2,y2=pos[v]
                
                # Determine if this is a GPU-NVSwitch link (2400 Gbps)
                is_gpu_nvswitch = (u < 32 and v in L1) or (v < 32 and u in L1)
                
                if is_gpu_nvswitch and bw >= 2400:
                    # GPU-NVSwitch links: solid line, thicker, darker
                    ls = '-'
                    lw = 2.5
                    alpha = 0.3
                    color = 'black'
                elif bw >= 400:
                    # Other high-bandwidth links (GPU-L2, L2-Spine): dashed line
                    ls = '--'
                    lw = 2.0
                    alpha = 0.4
                    color = 'blue'
                else:
                    # Degraded links: dotted line, red
                    ls = ':'
                    lw = 2.5
                    alpha = 0.5
                    color = 'red'
                
                plt.plot([x1,x2],[y1,y2], ls, linewidth=lw, color=color, alpha=alpha, zorder=0)
    groupA = [g for g in gpus if 0<=g<=15]
    groupB = [g for g in gpus if 16<=g<=31]
    
    # Create custom legend entries for link types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, linestyle='-', alpha=0.3, label='GPU↔NVSwitch (2400 Gbps)'),
        Line2D([0], [0], color='blue', linewidth=2.0, linestyle='--', alpha=0.4, label='GPU↔Rail, Rail↔Spine (400 Gbps)'),
    ]
    
    plt.scatter([pos[g][0] for g in groupA],[pos[g][1] for g in groupA], marker='s', s=400, label="Group A (0–15)")
    plt.scatter([pos[g][0] for g in groupB],[pos[g][1] for g in groupB], marker='^', s=400, label="Group B (16–31)")
    plt.scatter([pos[s][0] for s in L1],[pos[s][1] for s in L1], marker='o', s=400, label="NVSwitch")
    plt.scatter([pos[s][0] for s in L2],[pos[s][1] for s in L2], marker='D', s=400, label="Rail switch")
    plt.scatter([pos[spine][0]],[pos[spine][1]], marker='^', s=400, label="Spine")
    for g in gpus:
        x,y = pos[g]
        plt.text(x, y-0.1, str(g), ha='center', va='top', fontsize=font_size)
    for s in L2+[spine]:
        x,y = pos[s]
        plt.text(x+1.2, y-0.08, str(s), ha='center', va='bottom', fontsize=font_size, color='darkgreen')
    for s in L1:
        x,y = pos[s]
        plt.text(x+1, y-0.08, str(s), ha='center', va='bottom', fontsize=font_size, color='darkgreen')
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
