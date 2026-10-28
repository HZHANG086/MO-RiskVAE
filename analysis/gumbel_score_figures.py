"""
Score analysis figures for Gumbel-Softmax VAE.
Generates: risk score distribution, risk score rank plot.
"""
import sys, torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/work/zh/gumbel-softmax-vae/myevae/src")
from myevae.nn import MultiModalVAE
from myevae.dataset import Dataset
from torch.nn import LeakyReLU, Tanh
from torch.utils.data import DataLoader

valid_feat = pd.read_csv("/work/zh/myevae/inputs/valid_features_os_processed.csv", index_col=0)
valid_lab = pd.read_csv("/work/zh/myevae/inputs/valid_labels.csv", index_col=0)[["censos", "oscdy"]]
valid_df = pd.concat([valid_lab, valid_feat], axis=1)

input_types = ['exp','cna','gistic','fish','sbs','ig']
input_types_sub = ['clin']
all_types = input_types + input_types_sub
event = valid_df["censos"].values
duration = valid_df["oscdy"].values

# Load Gumbel model (0.7788 from output/)
pth = "/work/zh/gumbel-softmax-vae/myevae/output/test_model_os.pth"
ds = Dataset(valid_df, all_types, event_indicator_col="censos", event_time_col="oscdy", device="cpu")
input_dims = [getattr(ds, f"X_{t}").shape[1] for t in input_types]
input_dims_sub = [getattr(ds, f"X_{t}").shape[1] for t in input_types_sub]

model = MultiModalVAE(
    input_types=input_types, input_dims=input_dims,
    layer_dims=[[32,4],[16,4],[4,1],[4,1],[4,1],[4,1]],
    input_types_subtask=input_types_sub, input_dims_subtask=input_dims_sub,
    layer_dims_subtask=[4,1], z_dim=32, K=4,
    activation=LeakyReLU(0.01), subtask_activation=Tanh(),
    dropout=0.0, dropout_subtask=0.0
)
model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
model.eval()

loader = DataLoader(ds, batch_size=1024, shuffle=False)
risks = []
for _, data in enumerate(loader):
    with torch.no_grad():
        iv = [data[f'X_{t}'] for t in input_types]
        it = [data[f'X_{t}'] for t in input_types_sub]
        _, _, _, _, rp = model.forward((iv, it))
        risks.append(rp.flatten().numpy())
risks = np.concatenate(risks)

# Save risk scores
out_df = pd.DataFrame({
    "risk_score": risks,
    "event": event,
    "duration": duration,
}, index=valid_df.index)
out_df.to_csv("/work/zh/fig_score_risks_valid.csv")
print("[OK] Saved: /work/zh/fig_score_risks_valid.csv")

# Figure 1: Risk score distribution by event
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Histogram by event
ax = axes[0]
ax.hist(risks[event==0], bins=20, alpha=0.6, label="Censored", color="#5B9BD5", edgecolor="white")
ax.hist(risks[event==1], bins=20, alpha=0.6, label="Event", color="#C00000", edgecolor="white")
ax.set_xlabel("Risk score", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("(a) Risk score distribution by event status", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (b) Risk score rank plot (sorted by risk, colored by event)
ax = axes[1]
order = np.argsort(risks)
x = np.arange(len(risks))
ax.scatter(x, risks[order], c=event[order], cmap="RdYlGn_r", s=15, alpha=0.8, edgecolors="none")
ax.set_xlabel("Sample rank (by risk, low→high)", fontsize=11)
ax.set_ylabel("Risk score", fontsize=11)
ax.set_title("(b) Risk scores ranked (green=censored, red=event)", fontsize=12)
ax.grid(True, alpha=0.2)

fig.suptitle("Gumbel-Softmax VAE: Risk score analysis (OS, validation, C-index=0.7788)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("/work/zh/fig_score_gumbel.png", dpi=300, bbox_inches="tight")
plt.close()
print("[OK] Saved: /work/zh/fig_score_gumbel.png")
