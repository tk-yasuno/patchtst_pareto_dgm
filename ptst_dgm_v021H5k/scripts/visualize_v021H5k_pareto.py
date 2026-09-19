"""
visualize_v021H5k_pareto.py
Pareto frontier visualization for v0.2.1H5k (9D horizon-specific experiment).
Reads ptst_archive_v021H5k_log.jsonl and generates figures for the paper.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
LOG_FILE  = WORKSPACE / "ptst_dgm_v021H5k/results/ptst_archive_v021H5k_log.jsonl"
OUT_DIR   = WORKSPACE / "paper_patchtst_dgm/2_Main/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["30d", "60d", "90d"]
COLORS   = {"30d": "#e74c3c", "60d": "#2980b9", "90d": "#27ae60"}

# ── Load ──────────────────────────────────────────────────────────────────────
all_trials, pareto = [], []
with open(LOG_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            all_trials.append(d)
            if d.get("is_pareto_optimal"):
                pareto.append(d)

print(f"Total trials: {len(all_trials)}, Pareto-optimal: {len(pareto)}")

best_f1  = max(pareto, key=lambda s: s["macro_f1"])
best_fpr = min(pareto, key=lambda s: s["mean_fpr"])
print(f"Best macro-F1 : {best_f1['macro_f1']:.4f}  (trial {best_f1['trial_number']})")
print(f"Best mean-FPR : {best_fpr['mean_fpr']:.4f}  (trial {best_fpr['trial_number']})")

# ── Figure 1: macro-F1 vs mean-FPR ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
x = [s["mean_fpr"]  for s in pareto]
y = [s["macro_f1"]  for s in pareto]
t = [s["trial_number"] for s in pareto]

sc = ax.scatter(x, y, s=40, alpha=0.55, c=t, cmap="viridis",
                edgecolors="none", zorder=3, label="Pareto solutions")
ax.scatter(best_f1["mean_fpr"],  best_f1["macro_f1"],
           s=250, marker="*", color="#e74c3c", edgecolors="black", linewidth=1,
           zorder=5, label=f"Best F1 #{best_f1['trial_number']} ({best_f1['macro_f1']:.3f})")
ax.scatter(best_fpr["mean_fpr"], best_fpr["macro_f1"],
           s=180, marker="s", color="#2980b9", edgecolors="black", linewidth=1,
           zorder=5, label=f"Best FPR #{best_fpr['trial_number']} ({best_fpr['mean_fpr']:.3f})")

# sort by FPR for frontier line
sx = sorted(zip(x, y))
ax.plot([v[0] for v in sx], [v[1] for v in sx], "k--", lw=0.8, alpha=0.4, zorder=2)

ax.set_xlabel("Mean FPR", fontsize=11)
ax.set_ylabel("Macro F1", fontsize=11)
ax.set_title("Pareto Frontier — v0.2.1H5k (3{,}000 iterations)\n9D Horizon-Specific Focal Loss DGM",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="Trial number")
plt.tight_layout()
out1 = OUT_DIR / "v021H5k_pareto_macro_f1_vs_fpr.png"
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: per-horizon F1 vs FPR (3 subplots) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, h in zip(axes, HORIZONS):
    xh = [s["objectives"][f"fpr_{h}"] for s in pareto]
    yh = [s["objectives"][f"f1_{h}"]  for s in pareto]
    th = [s["trial_number"]            for s in pareto]
    sc = ax.scatter(xh, yh, s=30, alpha=0.5, c=th, cmap="plasma",
                    edgecolors="none", zorder=3)
    bi = int(np.argmax(yh))
    ax.scatter(xh[bi], yh[bi], s=200, marker="*", color="#e74c3c",
               edgecolors="black", linewidth=0.8, zorder=5,
               label=f"Best F1 #{th[bi]}")
    sx2 = sorted(zip(xh, yh))
    ax.plot([v[0] for v in sx2], [v[1] for v in sx2], "k--", lw=0.7, alpha=0.35)
    ax.set_xlabel(f"FPR ({h})", fontsize=10)
    ax.set_ylabel(f"F1 ({h})", fontsize=10)
    ax.set_title(f"Horizon {h}", fontsize=11, fontweight="bold", color=COLORS[h])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Per-Horizon Pareto Frontier — v0.2.1H5k (3,000 iters)",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
out2 = OUT_DIR / "v021H5k_pareto_per_horizon_f1_fpr.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: horizon-specific parameter distributions (violin) ───────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
param_sets = {
    "focal\_alpha": [f"focal_alpha_{h}" for h in HORIZONS],
    "focal\_gamma": [f"focal_gamma_{h}" for h in HORIZONS],
    "w\_normal":    [f"w_normal_{h}"    for h in HORIZONS],
}
for ax, (plabel, pkeys) in zip(axes, param_sets.items()):
    data   = [[s["params"][k] for s in pareto] for k in pkeys]
    colors = [COLORS[h] for h in HORIZONS]
    vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                       showextrema=True)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(HORIZONS, fontsize=10)
    ax.set_title(f"${plabel}$ distribution", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("Horizon-Specific Parameter Distributions (Pareto solutions, 3k iters)",
             fontsize=10, fontweight="bold", y=1.01)
plt.tight_layout()
out3 = OUT_DIR / "v021H5k_param_distributions.png"
plt.savefig(out3, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── Figure 4: macro-F1 convergence over iterations ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
iters  = [d["trial_number"] for d in all_trials]
f1vals = [d["macro_f1"]     for d in all_trials]
# running best
best_so_far = []
cur_best = 0.0
for v in f1vals:
    cur_best = max(cur_best, v)
    best_so_far.append(cur_best)

ax.scatter(iters, f1vals, s=4, alpha=0.25, color="gray", label="All trials")
ax.plot(iters, best_so_far, color="#e74c3c", lw=1.8, label="Running best")
ax.axhline(best_f1["macro_f1"], color="#e74c3c", ls="--", lw=1,
           label=f"Best {best_f1['macro_f1']:.4f} @ iter {best_f1['trial_number']}")
ax.axhline(0.656, color="orange", ls=":", lw=1, label="8D baseline (0.656)")
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("Macro F1", fontsize=11)
ax.set_title("Macro-F1 Convergence — v0.2.1H5k (3,000 iters)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out4 = OUT_DIR / "v021H5k_convergence.png"
plt.savefig(out4, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")

print("\nAll figures generated.")
