"""
Figure: Loss component magnitude comparison across ablation variants.
Stacked/grouped bar chart showing survival, reconstruction, and regularization losses.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

variants = ["A0\nKL (β=1)", "A1\nβ-KL (β=0.33)", "A2\nMMD", "A3\nMMD+HSIC"]

surv =  [2220.30, 2171.38, 2067.60, 2077.32]
recon = [2.61, 2.65, 2.63, 2.68]
reg =   [0.25, 10.05, 0.002, 0.002]
hsic =  [0, 0, 0, 0.000162]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.3]})

# Left: absolute magnitudes (log scale)
colors = ["#C00000", "#5B9BD5", "#70AD47", "#FFC000"]
x = np.arange(len(variants))
width = 0.18

bars_surv = ax1.bar(x - 1.5*width, surv, width, color=colors[0], label="Survival (Cox)")
bars_recon = ax1.bar(x - 0.5*width, recon, width, color=colors[1], label="Reconstruction")
bars_reg = ax1.bar(x + 0.5*width, reg, width, color=colors[2], label="Regularization\n(KL / MMD)")
bars_hsic_vals = [h if h > 0 else 1e-6 for h in hsic]
bars_hsic = ax1.bar(x + 1.5*width, bars_hsic_vals, width, color=colors[3], label="HSIC")

for bar_group in [bars_hsic]:
    for j, bar in enumerate(bar_group):
        if hsic[j] == 0:
            bar.set_alpha(0.15)

ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=9)
ax1.set_ylabel("Loss magnitude (log scale)", fontsize=11)
ax1.set_title("(a) Loss Component Magnitudes", fontsize=11)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, axis="y", alpha=0.2)
ax1.set_ylim(1e-5, 1e4)

# Right: table
col_labels = ["A0 (KL)", "A1 (β-KL)", "A2 (MMD)", "A3 (MMD+HSIC)"]
row_labels = ["Survival loss", "Reconstruction", "Reg (KL/MMD) raw",
              "Reg weighted", "HSIC (×0.1)", "Reg / Surv ratio"]

cell_text = [
    ["~2220", "~2171", "~2068", "~2077"],
    ["2.61", "2.65", "2.63", "2.68"],
    ["0.25", "30.5", "0.002", "0.002"],
    ["0.25", "10.0", "0.002", "0.002"],
    ["—", "—", "—", "0.0002"],
    ["1.1×10⁻⁴", "4.6×10⁻³", "9.5×10⁻⁷", "9.5×10⁻⁷"],
]

ax2.axis("off")
table = ax2.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == -1:
        cell.set_facecolor("#D9E2F3")
        cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")

ax2.set_title("(b) Loss Decomposition (seed=42, training set)", fontsize=11, pad=20)

plt.tight_layout()
plt.savefig("/work/zh/fig_loss_scale.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] Saved: /work/zh/fig_loss_scale.png")
