import shap
from matplotlib import pyplot as plt

def summary_plot(shap_values, shap_data, plot_idx, featurenames, outputdir, title, filename):
    # ensure numpy for shap summary_plot (data may be tensors on device)
    if hasattr(shap_values, 'cpu'):
        shap_values = shap_values.cpu().numpy()
    if hasattr(shap_data, 'cpu'):
        shap_data = shap_data.cpu().numpy()
    fig = shap.summary_plot(shap_values[:,plot_idx,0],
                            shap_data[:,plot_idx],
                            featurenames[plot_idx],
                            max_display=10,
                            plot_size=(6,4),
                            alpha=0.5,
                            show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{outputdir}/{filename}', dpi=150)