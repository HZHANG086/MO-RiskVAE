"""
Extract data for Section 4.2 (best-value comparison).
"""
import sys, torch, numpy as np, pandas as pd
from scipy.stats import chi2

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

def kaplan_meier(dur, ev):
    order = np.argsort(dur)
    t, e = dur[order], ev[order]
    unique_times = np.unique(t[e == 1])
    surv = 1.0
    times, probs = [0], [1.0]
    for ut in unique_times:
        ar = np.sum(t >= ut)
        died = np.sum((t == ut) & (e == 1))
        if ar > 0:
            surv *= (1 - died / ar)
        times.append(ut)
        probs.append(surv)
    return np.array(times), np.array(probs)

def logrank(dur1, ev1, dur2, ev2):
    at = np.unique(np.concatenate([dur1[ev1==1], dur2[ev2==1]]))
    O1, E1 = 0, 0
    for t in at:
        n1, n2 = np.sum(dur1 >= t), np.sum(dur2 >= t)
        d1 = np.sum((dur1 == t) & (ev1 == 1))
        n, d = n1 + n2, d1 + np.sum((dur2 == t) & (ev2 == 1))
        if n > 0:
            O1 += d1
            E1 += n1 * d / n
    if E1 == 0:
        return 0, 1.0
    stat = (O1 - E1) ** 2 / E1
    return stat, 1 - chi2.cdf(stat, df=1)

# Load Gumbel 0.7788 model
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

# KM for 0.7788 model
median_risk = np.median(risks)
high = risks >= median_risk
low = ~high
_, p = logrank(duration[high], event[high], duration[low], event[low])
O_hi = np.sum(event[high])
O_lo = np.sum(event[low])
rate_hi = O_hi / max(np.sum(duration[high]), 1)
rate_lo = O_lo / max(np.sum(duration[low]), 1)
hr = rate_hi / rate_lo if rate_lo > 0 else float('inf')

print("=" * 60)
print("① Best C-index (单次训练) per model")
print("=" * 60)
best = {
    "A0 KL (β=1.0)": 0.7138,
    "A1 β-KL (β=0.33)": 0.7419,
    "A2 MMD": 0.7121,
    "A3 MMD+HSIC": 0.7140,
    "A4 Gumbel-Softmax": 0.7788,
}
for k, v in sorted(best.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")
print(f"\n★ Best: A4 Gumbel-Softmax = 0.7788")

print("\n" + "=" * 60)
print("② Gumbel best model (0.7788) — KM results")
print("=" * 60)
print(f"  log-rank p-value: {p:.6f}")
print(f"  HR: {hr:.2f}")
print(f"  n_high: {np.sum(high)}, n_low: {np.sum(low)}")
