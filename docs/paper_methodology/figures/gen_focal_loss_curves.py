"""
Generate Focal Loss curves for 3 horizons (best trial: trial_1865, iter_1866).

Best trial parameters:
  30d: alpha=0.614, gamma=1.974
  60d: alpha=0.863, gamma=1.680
  90d: alpha=0.709, gamma=0.779
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Best trial parameters (trial_1865, iter_1866, macro_F1=0.7685)
PARAMS = {
    "30d": {"alpha": 0.614, "gamma": 1.974, "color": "#e74c3c", "ls": "-"},
    "60d": {"alpha": 0.863, "gamma": 1.680, "color": "#2980b9", "ls": "--"},
    "90d": {"alpha": 0.709, "gamma": 0.779, "color": "#27ae60", "ls": "-."},
}

p_t = np.linspace(1e-4, 1.0 - 1e-4, 500)


def focal_loss(p_t, alpha, gamma):
    """FL for the anomaly class (y=1): alpha_y,h = alpha."""
    return alpha * (1 - p_t) ** gamma * (-np.log(p_t))


fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)

for ax, (horizon, p) in zip(axes, PARAMS.items()):
    fl = focal_loss(p_t, p["alpha"], p["gamma"])
    # Reference: CE and standard FL (gamma=2, alpha=0.25)
    ce = -np.log(p_t)
    fl_ref = 0.25 * (1 - p_t) ** 2 * (-np.log(p_t))

    ax.plot(p_t, ce,     color="gray",   ls=":", lw=1.2, label=r"CE ($\alpha$=1, $\gamma$=0)")
    ax.plot(p_t, fl_ref, color="orange", ls=":", lw=1.2, label=r"FL std ($\alpha$=0.25, $\gamma$=2)")
    ax.plot(p_t, fl,     color=p["color"], ls=p["ls"], lw=2.2,
            label=fr"FL {horizon} ($\alpha$={p['alpha']}, $\gamma$={p['gamma']})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel(r"$p_t$ (predicted prob. of correct class)", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9) if ax == axes[0] else None
    ax.set_title(f"Horizon {horizon}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark the alpha parameter (modulation at p_t=1 is 0, annotate peak focus)
    ax.axvline(x=0.5, color="black", ls=":", lw=0.8, alpha=0.5)

fig.suptitle(
    "Horizon-Specific Focal Loss Curves — Best Trial (macro-F1 = 0.769, iter 1866)",
    fontsize=10, fontweight="bold", y=1.01
)
plt.tight_layout()

out = "v021H5k_focal_loss_curves.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
