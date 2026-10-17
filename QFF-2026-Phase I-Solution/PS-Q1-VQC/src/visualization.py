from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

def plot_coupling_graph(graph, title, output_path):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    pos=nx.spring_layout(graph,seed=42)
    plt.figure(figsize=(6,4)); nx.draw_networkx(graph,pos=pos,with_labels=True,node_size=650)
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(output_path,dpi=180); plt.close()

def plot_training_loss(history, output_path):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(6,4)); plt.plot(range(1,len(history)+1),history,marker="o",markersize=3)
    plt.xlabel("Optimizer callback step"); plt.ylabel("Binary cross-entropy")
    plt.title("VQC training convergence"); plt.tight_layout(); plt.savefig(output_path,dpi=180); plt.close()

def plot_depth_comparison(rows, output_path):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(6,4)); plt.bar([r["processor"] for r in rows],[r["depth"] for r in rows])
    plt.ylabel("Compiled circuit depth"); plt.title("Processor A vs Processor B")
    plt.tight_layout(); plt.savefig(output_path,dpi=180); plt.close()
