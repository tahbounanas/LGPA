# LGPA — Log-Gravity Propagation Algorithm

A deterministic, tuning-free community detection algorithm for complex networks, implemented in C++ (via [pybind11](https://github.com/pybind/pybind11)) with a simple Python/[NetworkX](https://networkx.org/) interface.

LGPA resolves two well-known weaknesses of standard Label Propagation — run-to-run instability and the formation of oversized *"monster"* communities — using a Laplacian-smoothed Jaccard similarity, a statistical Coring phase, and a log-gravity propagation rule that logarithmically dampens the influence of high-degree hubs. Its threshold and update rule are derived entirely from the graph's own structure, so there is nothing to tune, and its native C++ core scales to networks of tens of thousands of nodes in seconds.

This repository is the reference implementation for the paper *"LGPA: Log-Gravity Propagation Algorithm for Community Detection in Complex Networks"* (ASONAM 2026, Research Track). See [Citation](#citation).

---

## Requirements

Installing LGPA compiles a small C++ extension, so you need a C++ compiler in addition to Python. Everything else is installed automatically.

| Requirement | Notes |
|---|---|
| **Python** | 3.8 or newer |
| **A C++ compiler** | Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select the *"Desktop development with C++"* workload). macOS: `xcode-select --install`. Linux: `sudo apt install build-essential` (or your distro's equivalent). |
| **pybind11** | Installed automatically during the build. |
| **networkx** | Installed automatically as a dependency. |

---

## Installation

### Option 1 — Install directly from GitHub (recommended)

```bash
pip install git+https://github.com/tahbounanas/LGPA.git
```

### Option 2 — Clone, then install

```bash
git clone https://github.com/tahbounanas/LGPA.git
cd LGPA
pip install .
```

To develop (edit the C++/Python and rebuild in place), use `pip install -e .` instead.

`pip` handles `pybind11`, `networkx`, and compiling the extension. After it finishes, LGPA is importable from any folder in that Python environment.

---

## Usage

```python
import networkx as nx
from LGPA import LGPA

# A simple graph with two communities of 10 nodes each,
# joined by a single bridge edge.
G = nx.Graph()
for start in (0, 10):
    block = range(start, start + 10)
    for i in block:
        for j in block:
            if i < j:
                G.add_edge(i, j)   # dense links inside each community
G.add_edge(9, 10)                  # one bridge between the two communities

# Run LGPA
lgpa = LGPA(G)
partition = lgpa.fit_predict(max_iter=50) # or simply   partition = LGPA(G).fit_predict()


# partition: {node -> community_id}
print("Communities found:", len(set(partition.values())))
print(partition)
```

Expected output:

```
Communities found: 2
{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1}
```

| Input graph | LGPA result |
|:---:|:---:|
| ![Input graph](lgpa_before.jpg) | ![LGPA communities](lgpa_after.jpg) |

LGPA correctly separates the two communities (nodes 0–9 and 10–19) despite the bridge edge linking them.

`fit_predict` returns a dictionary mapping each node to its community id, so it works with any node labels (integers, strings, etc.), not just consecutive integers. LGPA is fully deterministic: the same graph always yields the same partition, with no random seed.

**Parameters**

- `max_iter` (int, default `50`): a safeguard cap on the number of propagation sweeps. It is not a tuned parameter — the loop stops on its own once labels stabilize, which in practice happens well before this cap.

---

## Example on a real dataset (Thiers)

The [`Datasets/`](Datasets/) folder contains the **Thiers** high-school contact network (327 nodes, 9 ground-truth classes): `Thiers.gml` is the graph and `Thiers_GR.txt` holds the ground-truth class label of each node (in GML node order).

> This example also uses `scikit-learn` and `scipy` for the metrics (`pip install scikit-learn scipy`); they are not required by LGPA itself.

```python
import json
import time
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score as nmi_score,
    adjusted_rand_score as ari_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np

from LGPA import LGPA

# Load the graph and its ground-truth labels
G = nx.read_gml("Datasets/Thiers.gml", label="id").to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))
gt_labels = json.load(open("Datasets/Thiers_GR.txt"))

nodes = list(G.nodes())
gt = {nodes[i]: gt_labels[i] for i in range(len(nodes))}   # GR is in GML node order

# Run LGPA (timed)
start = time.perf_counter()
partition = LGPA(G).fit_predict()
runtime = time.perf_counter() - start

# Encode labels as integers
classes = sorted(set(gt.values()))
cmap = {c: i for i, c in enumerate(classes)}
y_true = np.array([cmap[gt[n]] for n in nodes])
y_pred = np.array([partition[n] for n in nodes])

# NMI and ARI
nmi = nmi_score(y_true, y_pred)
ari = ari_score(y_true, y_pred)

# Macro-F1 (align predicted communities to ground-truth classes via Hungarian matching)
labels_p = sorted(set(y_pred))
C = confusion_matrix(y_true, [labels_p.index(x) for x in y_pred])
n = max(C.shape)
Cp = np.zeros((n, n)); Cp[:C.shape[0], :C.shape[1]] = C
r, c = linear_sum_assignment(-Cp)
mapping = {labels_p[cc]: rr for rr, cc in zip(r, c) if cc < len(labels_p)}
y_pred_aligned = np.array([mapping.get(x, -1) for x in y_pred])
f1 = f1_score(y_true, y_pred_aligned, average="macro")

print(f"Communities found: {len(set(y_pred))}")
print(f"NMI: {nmi:.3f}")
print(f"ARI: {ari:.3f}")
print(f"F1 : {f1:.3f}")
print(f"Runtime: {runtime:.3f} s")
```

Output:

```
Communities found: 9
NMI: 0.970
ARI: 0.964
F1 : 0.979
Runtime: 0.116 s
```

LGPA recovers all 9 classes with near-perfect agreement to the ground truth (NMI 0.970, ARI 0.964). In the two coloured figures below, each detected community has been matched to its best-corresponding ground-truth class (via Hungarian assignment) and drawn in that class's colour, so the ground-truth and LGPA plots line up directly. The metrics, not the colours, are what quantify the agreement.

| Input network | Ground truth | LGPA communities |
|:---:|:---:|:---:|
| ![Thiers input](thiers_before.jpg) | ![Thiers ground truth](thiers_groundtruth.jpg) | ![Thiers LGPA](thiers_after.jpg) |


### Reproducing the figures

The three plots above are produced with the snippet below (requires `matplotlib`, `pip install matplotlib`). A single shared layout is used so the input, ground-truth, and LGPA figures line up node-for-node.

```python
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from LGPA import LGPA

# Load graph + ground truth
G = nx.read_gml("Datasets/Thiers.gml", label="id").to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))
gt_labels = json.load(open("Datasets/Thiers_GR.txt"))
nodes = list(G.nodes())
gt = {nodes[i]: gt_labels[i] for i in range(len(nodes))}

# Run LGPA
partition = LGPA(G).fit_predict()

# One shared layout so all three plots are comparable
pos = nx.spring_layout(G, seed=42, k=0.3, iterations=60)
cmap = plt.colormaps["tab10"]

def draw(color_by, title, filename, legend=None):
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=color_by, edgecolors="#333333",
                           linewidths=0.4, node_size=90)
    if legend:
        plt.legend(handles=legend, loc="upper left", fontsize=8)
    plt.title(title, fontsize=14)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight"); plt.close()

# 1) Input graph (grey, unlabelled)
draw("#cccccc", "Thiers network - input", "thiers_before.jpg")

# 2) Ground truth (coloured by true class, with legend)
classes = sorted(set(gt.values()))
gt_colors = [cmap(classes.index(gt[n])) for n in G.nodes()]
handles = [mpatches.Patch(color=cmap(i), label=c) for i, c in enumerate(classes)]
draw(gt_colors, f"Thiers - ground truth ({len(classes)} classes)",
     "thiers_groundtruth.jpg", legend=handles)

# 3) LGPA result — colours matched to ground-truth classes via Hungarian assignment
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

y_true = np.array([classes.index(gt[n]) for n in nodes])
y_pred = np.array([partition[n] for n in nodes])
labels_p = sorted(set(y_pred))
C = confusion_matrix(y_true, [labels_p.index(x) for x in y_pred])
m = max(C.shape); Cp = np.zeros((m, m)); Cp[:C.shape[0], :C.shape[1]] = C
r, c = linear_sum_assignment(-Cp)
comm_to_class = {labels_p[cc]: rr for rr, cc in zip(r, c) if cc < len(labels_p)}

lgpa_colors = [cmap(comm_to_class.get(partition[n], len(classes))) for n in G.nodes()]
draw(lgpa_colors, f"Thiers - LGPA ({len(set(y_pred))} communities)",
     "thiers_after.jpg", legend=handles)
```

---

## Method at a glance

1. **Preprocessing** — a Laplacian-smoothed Jaccard similarity is computed for every edge, per-node structural strength is aggregated, and an adaptive threshold is derived from the graph's Structural Complexity Index.
2. **Phase 1 (Coring)** — nodes are merged with their most similar neighbour, in a deterministic strength-based order, to form stable proto-communities.
3. **Phase 2 (Log-Gravity Propagation)** — remaining labels are updated with a log-gravity score in which each neighbour's influence grows only logarithmically with its strength, suppressing hub dominance and preventing the avalanche effect.

---

## Citation

If you use LGPA in your research, please cite:

```bibtex

```

---

## License

Released under the [MIT License](LICENSE).