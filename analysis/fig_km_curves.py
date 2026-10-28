"""
Figure: Kaplan-Meier survival curves for all ablation variants (seed=42).
Uses median risk score to split high/low risk.
Implements KM + log-rank + HR using only numpy/scipy (no lifelines needed).
"""
import sys, json, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

def kaplan_meier(durations, events):
    """Return (times, survival_probs) for KM curve."""
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    unique_times = np.unique(t[e == 1])
    
    n = len(t)
    surv = 1.0
    times = [0]
    probs = [1.0]
    
    for ut in unique_times:
        at_risk = np.sum(t >= ut)
        died = np.sum((t == ut) & (e == 1))
        if at_risk > 0:
            surv *= (1 - died / at_risk)
        times.append(ut)
        probs.append(surv)
    
    return np.array(times), np.array(probs)

def logrank_test_fn(dur1, ev1, dur2, ev2):
    """Log-rank test. Returns chi2 statistic and p-value."""
    all_times = np.unique(np.concatenate([dur1[ev1 == 1], dur2[ev2 == 1]]))
    
    O1, E1 = 0, 0
    for t in all_times:
        n1 = np.sum(dur1 >= t)
        n2 = np.sum(dur2 >= t)
        d1 = np.sum((dur1 == t) & (ev1 == 1))
        d2 = np.sum((dur2 == t) & (ev2 == 1))
        n = n1 + n2
        d = d1 + d2
        if n > 0:
            O1 += d1
            E1 += n1 * d / n
    
    if E1 == 0:
        return 0, 1.0
    stat = (O1 - E1) ** 2 / E1
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p

def hazard_ratio_simple(dur, ev, group):
    """Simple Cox-like HR estimate using Mantel-Haenszel method."""
    all_times = np.unique(dur[ev == 1])
    numer, denom = 0, 0
    for t in all_times:
        at_risk_1 = np.sum((dur >= t) & (group == 1))
        at_risk_0 = np.sum((dur >= t) & (group == 0))
        d1 = np.sum((dur == t) & (ev == 1) & (group == 1))
        d0 = np.sum((dur == t) & (ev == 1) & (group == 0))
        n = at_risk_0 + at_risk_1
        d = d0 + d1
        if n > 0 and at_risk_0 > 0:
            numer += d1 - at_risk_1 * d / n
            denom += at_risk_0 * at_risk_1 * d * (n - d) / (n * n * (n - 1)) if n > 1 else 0
    if denom == 0:
        return float('inf')
    return np.exp(numer / (denom * n / at_risk_0)) if denom > 0 else float('inf')


valid_feat = pd.read_csv("/work/zh/myevae/inputs/valid_features_os_processed.csv", index_col=0)
valid_lab = pd.read_csv("/work/zh/myevae/inputs/valid_labels.csv", index_col=0)[["censos", "oscdy"]]
valid_df = pd.concat([valid_lab, valid_feat], axis=1)

input_types = ['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig']
input_types_sub = ['clin']
all_types = input_types + input_types_sub


def get_risk_predictions(codebase_src, pth_path, data_df):
    """Load model and predict risk scores."""
    saved_path = sys.path.copy()
    sys.path.insert(0, codebase_src)
    
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("myevae"):
            del sys.modules[mod_name]
    
    from myevae.nn import MultiModalVAE as Model
    from myevae.dataset import Dataset
    from torch.nn import LeakyReLU, Tanh
    
    dataset = Dataset(data_df, all_types, event_indicator_col="censos",
                      event_time_col="oscdy", device="cpu")
    input_dims = [getattr(dataset, f"X_{t}").shape[1] for t in input_types]
    input_dims_sub = [getattr(dataset, f"X_{t}").shape[1] for t in input_types_sub]
    
    model = Model(
        input_types=input_types, input_dims=input_dims,
        layer_dims=[[32, 4], [16, 4], [4, 1], [4, 1], [4, 1], [4, 1]],
        input_types_subtask=input_types_sub, input_dims_subtask=input_dims_sub,
        layer_dims_subtask=[4, 1], z_dim=32,
        activation=LeakyReLU(negative_slope=0.01), subtask_activation=Tanh(),
        dropout=0.0, dropout_subtask=0.0)
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    all_preds = []
    for _, data in enumerate(loader):
        with torch.no_grad():
            inputs_vae = [data[f'X_{t}'] for t in model.input_types_vae]
            inputs_task = [data[f'X_{t}'] for t in model.input_types_subtask]
            out = model.forward((inputs_vae, inputs_task))
            riskpred = out[-1]  # last return is riskpred (4 for standard, 5 for Gumbel)
            all_preds.append(riskpred.flatten().numpy())
    
    sys.path = saved_path
    return np.concatenate(all_preds)


variants = [
    ("A0: KL (β=1.0)", "/work/zh/myevae/src",
     "/work/zh/myevae/output/beta_1.0_s42/beta_1.0/test_model_os.pth"),
    ("A1: β-KL (β=0.33)", "/work/zh/myevae/src",
     "/work/zh/myevae/output/beta_0.33_s42/beta_0.33/test_model_os.pth"),
    ("A2: MMD", "/work/zh/mmd-vae/myevae/src",
     "/work/zh/mmd-vae/myevae/output_s42/default_os_os.pth"),
    ("A3: MMD+HSIC", "/work/zh/HSIC-myeVAE/myevae/src",
     "/work/zh/HSIC-myeVAE/myevae/output_s42/default_os.pth"),
    ("A4: Gumbel", "/work/zh/gumbel-softmax-vae/myevae/src",
     "/work/zh/gumbel-softmax-vae/myevae/output_s42/test_model_os.pth"),
]

event = valid_df["censos"].values.astype(int)
duration = valid_df["oscdy"].values.astype(float)
colors_hi = "#C00000"
colors_lo = "#5B9BD5"
suptitle = "Kaplan–Meier Survival Curves by Predicted Risk Group\n(OS, validation set, seed=42, median split)"
results_text = []
risk_cache = {}


def draw_one_panel(ax, name, src, pth, i, n_total):
    key = (src, pth)
    if key not in risk_cache:
        risk_cache[key] = get_risk_predictions(src, pth, valid_df)
    risks = risk_cache[key]
    median_risk = np.median(risks)
    high_mask = risks >= median_risk
    low_mask = ~high_mask
    t_hi, s_hi = kaplan_meier(duration[high_mask], event[high_mask])
    t_lo, s_lo = kaplan_meier(duration[low_mask], event[low_mask])
    ax.step(t_hi, s_hi, where="post", color=colors_hi, linewidth=2, label="High risk")
    ax.step(t_lo, s_lo, where="post", color=colors_lo, linewidth=2, label="Low risk")
    stat, p_val = logrank_test_fn(
        duration[high_mask], event[high_mask],
        duration[low_mask], event[low_mask])
    O_hi = np.sum(event[high_mask])
    O_lo = np.sum(event[low_mask])
    rate_hi = O_hi / np.sum(duration[high_mask]) if np.sum(duration[high_mask]) > 0 else 0
    rate_lo = O_lo / np.sum(duration[low_mask]) if np.sum(duration[low_mask]) > 0 else 0
    hr = rate_hi / rate_lo if rate_lo > 0 else float('inf')
    # MICCAI-style font sizes (readable in single/double column)
    p_str = f"p = {p_val:.4f}" if p_val >= 0.0001 else f"p < 0.0001"
    ax.set_title(f"{name}\nlog-rank {p_str}\nHR = {hr:.2f}", fontsize=12)
    ax.set_xlabel("Time (days)", fontsize=11)
    if i == 0:
        ax.set_ylabel("Survival probability", fontsize=11)
    ax.legend(loc="lower left", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    results_text.append(f"{name}: log-rank p={p_val:.6f}, HR={hr:.2f}, n_hi={np.sum(high_mask)}, n_lo={np.sum(low_mask)}")


# ----- Single combined figure: fig_km_curves.png (top 3 + bottom 2 centered) -----
# Larger figure + bigger horizontal/vertical gaps to avoid overlapping labels.
fig = plt.figure(figsize=(26, 13))
margin_left = 0.03
margin_right = 0.03
gap = 0.045   # horizontal gap between panels (normalized)
top_panel_w = (1 - margin_left - margin_right - 2 * gap) / 3
# Vertical: same panel height for both rows; leave gap between rows
row_h = 0.34   # same height for each row
top_y0, top_y1 = 0.54, 0.54 + row_h   # top row (0.54–0.88)
bot_y0, bot_y1 = 0.08, 0.08 + row_h   # bottom row same height (0.08–0.42)

# Top row: 3 panels with gaps
axes = []
for k in range(3):
    x0 = margin_left + k * (top_panel_w + gap)
    axes.append(fig.add_axes([x0, top_y0, top_panel_w, top_y1 - top_y0]))
# Bottom row: 2 panels centered, same panel width and gap
two_panel_total = 2 * top_panel_w + gap
bottom_margin = (1 - two_panel_total) / 2
for k in range(2):
    x0 = bottom_margin + k * (top_panel_w + gap)
    axes.append(fig.add_axes([x0, bot_y0, top_panel_w, bot_y1 - bot_y0]))

for ax in axes[1:]:
    ax.sharey(axes[0])

for i, (name, src, pth) in enumerate(variants):
    print(f"Processing {name}...")
    draw_one_panel(axes[i], name, src, pth, i, 5)

fig.suptitle(suptitle, fontsize=16, y=0.96)
plt.savefig("/work/zh/fig_km_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] Saved: /work/zh/fig_km_curves.png (5 panels)")

# ----- Merged: 1 figure with all curves (5 colors, solid=high risk, dashed=low risk) -----
km_data = []
for name, src, pth in variants:
    key = (src, pth)
    risks = risk_cache[key]
    median_risk = np.median(risks)
    high_mask = risks >= median_risk
    low_mask = ~high_mask
    t_hi, s_hi = kaplan_meier(duration[high_mask], event[high_mask])
    t_lo, s_lo = kaplan_meier(duration[low_mask], event[low_mask])
    km_data.append((name, t_hi, s_hi, t_lo, s_lo))

# 5 distinct colors for the 5 variants (colorblind-friendly)
variant_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
for i, (name, t_hi, s_hi, t_lo, s_lo) in enumerate(km_data):
    c = variant_colors[i % len(variant_colors)]
    ax1.step(t_hi, s_hi, where="post", color=c, linewidth=2, linestyle="-", label=f"{name} — High risk")
    ax1.step(t_lo, s_lo, where="post", color=c, linewidth=2, linestyle="--", label=f"{name} — Low risk")
ax1.set_xlabel("Time (days)", fontsize=11)
ax1.set_ylabel("Survival probability", fontsize=11)
ax1.set_title("Kaplan–Meier Survival Curves by Predicted Risk Group\n(OS, validation set, seed=42, median split)", fontsize=12)
ax1.legend(loc="lower left", fontsize=9, ncol=1)
ax1.tick_params(axis="both", labelsize=10)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(0, 1.05)
ax1.set_xlim(0, None)
plt.tight_layout()
plt.savefig("/work/zh/fig_km_curves_merged1.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] Saved: /work/zh/fig_km_curves_merged1.png (1 plot, all curves)")

# ----- Merged: 2 panels — left: all High risk (5 colors), right: all Low risk (5 colors) -----
fig2, (ax_hi, ax_lo) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for i, (name, t_hi, s_hi, t_lo, s_lo) in enumerate(km_data):
    c = variant_colors[i % len(variant_colors)]
    ax_hi.step(t_hi, s_hi, where="post", color=c, linewidth=2, label=name)
    ax_lo.step(t_lo, s_lo, where="post", color=c, linewidth=2, label=name)
ax_hi.set_xlabel("Time (days)", fontsize=11)
ax_hi.set_ylabel("Survival probability", fontsize=11)
ax_hi.set_title("High risk", fontsize=12)
ax_hi.legend(loc="lower left", fontsize=9)
ax_hi.tick_params(axis="both", labelsize=10)
ax_hi.grid(True, alpha=0.2)
ax_hi.set_ylim(0, 1.05)
ax_hi.set_xlim(0, None)
ax_lo.set_xlabel("Time (days)", fontsize=11)
ax_lo.set_title("Low risk", fontsize=12)
ax_lo.legend(loc="lower left", fontsize=9)
ax_lo.tick_params(axis="both", labelsize=10)
ax_lo.grid(True, alpha=0.2)
ax_lo.set_ylim(0, 1.05)
ax_lo.set_xlim(0, None)
fig2.suptitle("Kaplan–Meier by variant (OS, validation, seed=42, median split)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("/work/zh/fig_km_curves_merged2.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] Saved: /work/zh/fig_km_curves_merged2.png (2 panels: High | Low)")

print("")
for r in results_text:
    print(f"  {r}")
