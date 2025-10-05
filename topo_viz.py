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

def hop_bw(adj, spine, gpu_to_l2, g1, g2):
    l2_1, bw1 = gpu_to_l2[g1]
    l2_2, bw2 = gpu_to_l2[g2]
    bu = adj[l2_1][spine]; bd = adj[spine][l2_2]
    return min(bw1, bu, bd, bw2)

def plot_bundles_with_adjacent(out_path, header, adj, spine, L2, l2_to_gpus, l2_uplink, gpu_to_l2, group_size=16):
    cols = len(L2)
    max_rows = max(len(l2_to_gpus[l2]) for l2 in L2)
    fig, ax = plt.subplots(figsize=(min(22, 2+cols*1.4), 2.5 + max_rows*0.9))
    gpu_pos = {}
    for ci, l2 in enumerate(L2):
        upl = l2_uplink[l2]
        ax.add_patch(plt.Rectangle((ci, max_rows+0.6), 0.9, 0.6, fill=False, edgecolor='black'))
        ax.text(ci+0.45, max_rows+0.9, f"L2 {l2}\n↑{upl}G", ha='center', va='center', fontsize=9)
        for ri, (gid, gbw) in enumerate(l2_to_gpus[l2]):
            y = max_rows - ri
            gpu_pos[gid]=(ci+0.45, y)
            ax.text(ci+0.45, y, f"{gid}", ha='center', va='center', fontsize=9)
            if gbw<=200:
                ax.text(ci+0.45, y-0.32, f"{gbw}", ha='center', va='center', fontsize=8)
        ax.text(ci+0.45, 0.15, f"{len(l2_to_gpus[l2])} GPUs", ha='center', va='bottom', fontsize=8)
    max_gpu = max([g for l in l2_to_gpus.values() for g,_ in l]) if l2_to_gpus else -1
    total_gpus = max_gpu+1
    for start in range(0, total_gpus, group_size):
        ids = list(range(start, min(start+group_size, total_gpus)))
        if len(ids)<2: continue
        for i in range(len(ids)):
            g1 = ids[i]; g2 = ids[(i+1)%len(ids)]
            if (g1 not in gpu_pos) or (g2 not in gpu_pos): continue
            bw = hop_bw(adj, spine, gpu_to_l2, g1,g2)
            x1,y1 = gpu_pos[g1]; x2,y2 = gpu_pos[g2]
            ls = '-' if bw>=400 else '--'
            ax.plot([x1,x2],[y1,y2], linestyle=ls, linewidth=1.8)
            xm,ym=(x1+x2)/2,(y1+y2)/2
            ax.text(xm, ym+0.15, f"{bw}", ha='center', va='bottom', fontsize=8)
    ax.set_xlim(-0.2, cols+0.2)
    ax.set_ylim(0, max_rows+1.5)
    ax.set_xticks([i+0.45 for i in range(cols)]); ax.set_xticklabels([str(l2) for l2 in L2])
    ax.set_yticks([])
    ax.set_title(f"NIC bundles with adjacent-GPU ring bandwidths\n{' '.join(header)}\nLine: solid=400, dashed=200; labels show bottleneck Gbps")
    ax.axhline(max_rows+0.6, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")

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
    for u in adj:
        for v,bw in adj[u].items():
            if u<v:
                x1,y1=pos[u]; x2,y2=pos[v]
                # ls='-' if bw>=400 else '--'
                # lw=1.5 if bw>=400 else 2.5
                ls='-' 
                lw=2.
                plt.plot([x1,x2],[y1,y2], ls, linewidth=lw, color='black', alpha=0.3,zorder=0)
                # if bw<=200:
                #     xm=(x1+x2)/2; ym=(y1+y2)/2
                #     plt.text(xm, ym+0.05, str(bw), fontsize=7, ha='center', va='bottom')
    groupA = [g for g in gpus if 0<=g<=31]
    # groupB = [g for g in gpus if 16<=g<=31]
    plt.scatter([pos[g][0] for g in groupA],[pos[g][1] for g in groupA], marker='s', s=400, label="Group A (0–15)")
    # plt.scatter([pos[g][0] for g in groupB],[pos[g][1] for g in groupB], marker='^', s=200, label="Group B (16–31)")
    plt.scatter([pos[s][0] for s in L1],[pos[s][1] for s in L1], marker='o', s=400, label="L1 Switch")
    plt.scatter([pos[s][0] for s in L2],[pos[s][1] for s in L2], marker='D', s=400, label="L2 Switch")
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
    plt.yticks([0,1,2,3], ["L1","GPUs","L2","Spine"], fontsize=font_size+5)
    # plt.title(f"Topology layers with groups (IDs shown)\n{' '.join(header)}")
    plt.xlabel("Logical position (rank order)", fontsize=font_size+5)
    plt.grid(True, linestyle=":")
    # plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="SimAI topology visualizations")
    ap.add_argument("topo", help="path to topology file (e.g., topo_N8_M2.txt)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_b = sub.add_parser("bundles", help="Draw NIC-bundle grid with adjacent-GPU ring bandwidths")
    ap_b.add_argument("--group-size", type=int, default=16, help="ring group size (default 16)")
    ap_b.add_argument("--out", default="simai_topo_bundles.png")
    ap_g = sub.add_parser("groups", help="Draw layered topo with group overlays")
    ap_g.add_argument("--out", default="simai_topo_groups.png")
    args = ap.parse_args()
    header, edges = parse_topo(args.topo)
    adj, spine, L1, L2, l2_to_gpus, l2_uplink, gpu_to_l2 = build_graph(edges)
    if args.cmd == "bundles":
        plot_bundles_with_adjacent(args.out, header, adj, spine, L2, l2_to_gpus, l2_uplink, gpu_to_l2, group_size=args.group_size)
    elif args.cmd == "groups":
        plot_groups_layers(args.out, header, adj, spine, L1, L2)

if __name__ == "__main__":
    main()
