"""
Supplementary latent structure analysis for 4.3.
- β-KL (A1) vs Gumbel: μ t-SNE colored by risk (side-by-side)
- Spearman(PC1(μ), risk) and μ–risk alignment metrics
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

# Data
valid_feat = pd.read_csv("/work/zh/myevae/inputs/valid_features_os_processed.csv", index_col=0)
valid_lab = pd.read_csv("/work/zh/myevae/inputs/valid_labels.csv", index_col=0)[["censos", "oscdy"]]
valid_df = pd.concat([valid_lab, valid_feat], axis=1)
input_types = ['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig']
input_types_sub = ['clin']
all_types = input_types + input_types_sub

def load_myevae(codebase_src, pth_path):
    saved = sys.path.copy()
    sys.path.insert(0, codebase_src)
    for m in list(sys.modules.keys()):
        if m.startswith("myevae"):
            del sys.modules[m]
    from myevae.nn import MultiModalVAE as Model
    from myevae.dataset import Dataset
    from torch.nn import LeakyReLU, Tanh
    dataset = Dataset(valid_df, all_types, event_indicator_col="censos",
                      event_time_col="oscdy", device="cpu")
    input_dims = [getattr(dataset, f"X_{t}").shape[1] for t in input_types]
    input_dims_sub = [getattr(dataset, f"X_{t}").shape[1] for t in input_types_sub]
    model = Model(
        input_types=input_types, input_dims=input_dims,
        layer_dims=[[32,4],[16,4],[4,1],[4,1],[4,1],[4,1]],
        input_types_subtask=input_types_sub, input_dims_subtask=input_dims_sub,
        layer_dims_subtask=[4,1], z_dim=32,
        activation=LeakyReLU(0.01), subtask_activation=Tanh(),
        dropout=0.0, dropout_subtask=0.0
    )
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    sys.path = saved
    return model, dataset

def load_gumbel(pth_path):
    saved = sys.path.copy()
    sys.path.insert(0, "/work/zh/gumbel-softmax-vae/myevae/src")
    for m in list(sys.modules.keys()):
        if m.startswith("myevae"):
            del sys.modules[m]
    from myevae.nn import MultiModalVAE as Model
    from myevae.dataset import Dataset
    from torch.nn import LeakyReLU, Tanh
    dataset = Dataset(valid_df, all_types, event_indicator_col="censos",
                      event_time_col="oscdy", device="cpu")
    input_dims = [getattr(dataset, f"X_{t}").shape[1] for t in input_types]
    input_dims_sub = [getattr(dataset, f"X_{t}").shape[1] for t in input_types_sub]
    model = Model(
        input_types=input_types, input_dims=input_dims,
        layer_dims=[[32,4],[16,4],[4,1],[4,1],[4,1],[4,1]],
        input_types_subtask=input_types_sub, input_dims_subtask=input_dims_sub,
        layer_dims_subtask=[4,1], z_dim=32, K=4,
        activation=LeakyReLU(0.01), subtask_activation=Tanh(),
        dropout=0.0, dropout_subtask=0.0
    )
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    sys.path = saved
    return model, dataset

def extract_mu_risk_myevae(model, dataset):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    mus, risks = [], []
    for _, data in enumerate(loader):
        with torch.no_grad():
            iv = [data[f'X_{t}'] for t in model.input_types_vae]
            it = [data[f'X_{t}'] for t in model.input_types_subtask]
            _, mu, _, rp = model.forward((iv, it))
            mus.append(mu.numpy())
            risks.append(rp.flatten().numpy())
    return np.concatenate(mus), np.concatenate(risks)

def extract_mu_risk_gumbel(model, dataset):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    mus, risks = [], []
    for _, data in enumerate(loader):
        with torch.no_grad():
            iv = [data[f'X_{t}'] for t in model.input_types_vae]
            it = [data[f'X_{t}'] for t in model.input_types_subtask]
            _, mu, _, _, rp = model.forward((iv, it))
            mus.append(mu.numpy())
            risks.append(rp.flatten().numpy())
    return np.concatenate(mus), np.concatenate(risks)

def main():
    # Load A1 (β-KL) and Gumbel
    print("Loading A1 (β-KL)...")
    m_a1, ds_a1 = load_myevae(
        "/work/zh/myevae/src",
        "/work/zh/myevae/output/beta_0.33/test_model_os.pth"
    )
    mu_a1, risk_a1 = extract_mu_risk_myevae(m_a1, ds_a1)

    print("Loading Gumbel (0.7788)...")
    m_g, ds_g = load_gumbel("/work/zh/gumbel-softmax-vae/myevae/output/test_model_os.pth")
    mu_g, risk_g = extract_mu_risk_gumbel(m_g, ds_g)

    n = len(risk_a1)
    assert n == len(risk_g), "sample count mismatch"

    # 1) Spearman(PC1(μ), risk)
    pca_a1 = PCA(n_components=1).fit(mu_a1)
    pc1_a1 = pca_a1.transform(mu_a1).flatten()
    pca_g = PCA(n_components=1).fit(mu_g)
    pc1_g = pca_g.transform(mu_g).flatten()

    r_pc1_a1, p_pc1_a1 = spearmanr(pc1_a1, risk_a1)
    r_pc1_g, p_pc1_g = spearmanr(pc1_g, risk_g)

    # 2) Mean |Spearman(μ_i, risk)| across 32 dims
    rhos_a1 = []
    for j in range(mu_a1.shape[1]):
        r, _ = spearmanr(mu_a1[:, j], risk_a1)
        rhos_a1.append(np.nan if r is None else r)
    rhos_a1 = np.array(rhos_a1)
    mean_abs_rho_a1 = np.nanmean(np.abs(rhos_a1))
    max_abs_rho_a1 = np.nanmax(np.abs(rhos_a1))

    rhos_g = []
    for j in range(mu_g.shape[1]):
        r, _ = spearmanr(mu_g[:, j], risk_g)
        rhos_g.append(np.nan if r is None else r)
    rhos_g = np.array(rhos_g)
    mean_abs_rho_g = np.nanmean(np.abs(rhos_g))
    max_abs_rho_g = np.nanmax(np.abs(rhos_g))

    # 3) t-SNE of μ, colored by risk (same global color scale)
    print("t-SNE A1...")
    tsne_a1 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(mu_a1)
    print("t-SNE Gumbel...")
    tsne_g = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(mu_g)

    r_min = min(risk_a1.min(), risk_g.min())
    r_max = max(risk_a1.max(), risk_g.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    sc0 = ax.scatter(tsne_a1[:, 0], tsne_a1[:, 1], c=risk_a1, cmap="coolwarm", vmin=r_min, vmax=r_max, s=20, alpha=0.8, edgecolors="none")
    ax.set_title("A1 (β-KL): t-SNE of μ colored by risk", fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    sc1 = ax.scatter(tsne_g[:, 0], tsne_g[:, 1], c=risk_g, cmap="coolwarm", vmin=r_min, vmax=r_max, s=20, alpha=0.8, edgecolors="none")
    ax.set_title("Gumbel: t-SNE of μ colored by risk", fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.022, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label="Risk score")
    fig.suptitle("Latent μ vs risk: A1 (β-KL) vs Gumbel (same risk color scale)", fontsize=13, y=1.02)
    plt.savefig("/work/zh/fig_latent_risk_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: /work/zh/fig_latent_risk_comparison.png")

    # 4) Bar chart: Spearman PC1–risk and mean |ρ| (MICCAI-style: slate blue & teal)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["A1 (β-KL)", "Gumbel"]
    pc1_vals = [r_pc1_a1, r_pc1_g]
    mean_vals = [mean_abs_rho_a1, mean_abs_rho_g]
    x = np.arange(2)
    # MICCAI-style: restrained blue and teal (medical/neuro imaging common palette)
    colors_miccai = ["#3D5A80", "#4A90A4"]  # slate blue, teal
    axes[0].bar(x, pc1_vals, color=colors_miccai, edgecolor="none", linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("PC1(μ) vs risk")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(pc1_vals):
        axes[0].text(i, v + 0.02 if v >= 0 else v - 0.05, f"{v:.3f}", ha="center", fontsize=10)

    axes[1].bar(x, mean_vals, color=colors_miccai, edgecolor="none", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean |Spearman ρ|")
    axes[1].set_title("μ dimensions vs risk (mean over 32 dims)")
    for i, v in enumerate(mean_vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("/work/zh/fig_latent_risk_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: /work/zh/fig_latent_risk_correlation.png")

    # 5) Text summary for 4.3
    if abs(r_pc1_g) > abs(r_pc1_a1) and mean_abs_rho_g > mean_abs_rho_a1:
        conclusion_line = "  Gumbel shows stronger μ–risk alignment (higher |ρ| for PC1 and mean over dims); structural advantage is supported by this comparison."
    elif abs(r_pc1_g) < abs(r_pc1_a1) and mean_abs_rho_g < mean_abs_rho_a1:
        conclusion_line = "  A1 shows stronger μ–risk alignment; no evidence for Gumbel structural advantage."
    else:
        conclusion_line = "  Mixed: one metric favors Gumbel, one A1; no clear structural advantage."

    lines = [
        "Latent structure comparison: A1 (β-KL) vs Gumbel",
        "=" * 60,
        "Models: A1 = beta_0.33 (β-KL); Gumbel = output/ (C-index 0.7788).",
        "Validation set, same risk color scale in t-SNE.",
        "",
        "Spearman(PC1(μ), risk):",
        f"  A1 (β-KL):   ρ = {r_pc1_a1:.4f}  (p = {p_pc1_a1:.4e})",
        f"  Gumbel:      ρ = {r_pc1_g:.4f}  (p = {p_pc1_g:.4e})",
        "",
        "Mean |Spearman(μ_i, risk)| over 32 dimensions:",
        f"  A1 (β-KL):   {mean_abs_rho_a1:.4f}  (max |ρ| = {max_abs_rho_a1:.4f})",
        f"  Gumbel:      {mean_abs_rho_g:.4f}  (max |ρ| = {max_abs_rho_g:.4f})",
        "",
        "Interpretation:",
        "  - Higher |ρ| (PC1 or mean) => μ is more aligned with risk.",
        "  - If Gumbel > A1: Gumbel has stronger risk-aligned manifold.",
        "  - If similar: no evidence that Gumbel is structurally superior.",
        "",
        "Conclusion (for 4.3):",
        conclusion_line,
        "",
        "Figures:",
        "  fig_latent_risk_comparison.png  — t-SNE μ by risk (A1 vs Gumbel)",
        "  fig_latent_risk_correlation.png — PC1–risk and mean |ρ| bar chart",
    ]
    out = "\n".join(lines)
    with open("/work/zh/latent_structure_comparison.txt", "w") as f:
        f.write(out)
    print("[OK] Saved: /work/zh/latent_structure_comparison.txt")
    print()
    print(out)

if __name__ == "__main__":
    main()
