import shap
from matplotlib import pyplot as plt

def summary_plot(shap_values, shap_data, plot_idx, featurenames, outputdir, title, filename):
    
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