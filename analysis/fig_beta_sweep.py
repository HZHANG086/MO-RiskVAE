"""
Figure: Beta sensitivity curve (beta vs C-index).
1001 points from beta=0.0 to 1.0, step=0.001. Single unseeded runs.
Includes scatter + LOESS smoothed trend + best point annotation.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

base = "/work/zh/myevae/output"
betas, cindices = [], []

for d in sorted(glob.glob(os.path.join(base, "beta_*"))):
    dirname = os.path.basename(d)
    if "_s4" in dirname:
        continue
    try:
        beta_val = float(dirname.replace("beta_", ""))
    except ValueError:
        continue
    json_path = os.path.join(d, "test_model_os.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path) as f:
        r = json.load(f)
    betas.append(beta_val)
    cindices.append(r["best_epoch"]["valid_metric"])

betas = np.array(betas)
cindices = np.array(cindices)
order = np.argsort(betas)
betas = betas[order]
cindices = cindices[order]

smoothed = uniform_filter1d(cindices, size=31)

best_idx = np.argmax(cindices)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(betas, cindices, s=6, alpha=0.35, color="#5B9BD5", label="Single run (no fixed seed)", zorder=2)
ax.plot(betas, smoothed, color="#C00000", linewidth=2, label="Smoothed trend (window=31)", zorder=3)

ax.annotate(
    f"Best: β={betas[best_idx]:.2f}\nC={cindices[best_idx]:.4f}",
    xy=(betas[best_idx], cindices[best_idx]),
    xytext=(betas[best_idx] + 0.12, cindices[best_idx] + 0.02),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    fontsize=9, ha="left",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
)

ax.axhline(y=np.mean(cindices[betas > 0.9]), color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.text(0.02, np.mean(cindices[betas > 0.9]) + 0.005, "β→1 avg", fontsize=8, color="gray")

ax.set_xlabel("β (KL divergence weight)", fontsize=12)
ax.set_ylabel("Validation C-index (OS)", fontsize=12)
ax.set_title("β-VAE Sensitivity Analysis: Effect of KL Weight on Survival Prediction", fontsize=12)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0.42, 0.82)
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("/work/zh/fig_beta_sweep.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: /work/zh/fig_beta_sweep.png")
print(f"     Points: {len(betas)}")
print(f"     Best: beta={betas[best_idx]:.4f}, C-index={cindices[best_idx]:.4f}")
print(f"     beta>0.5 avg: {np.mean(cindices[betas>0.5]):.4f}")
print(f"     beta<0.1 avg: {np.mean(cindices[betas<0.1]):.4f}")
print(f"     beta 0.1-0.4 avg: {np.mean(cindices[(betas>=0.1)&(betas<=0.4)]):.4f}")
