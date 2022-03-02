"""Plot the metrics for the experiment log files"""
from pathlib import Path

from image_classification.data import load_logged_metrics, plot_metrics_lab03

logs_list = [
    ('Fixed lr=0.001)', './logs/w3_logged_metrics_fixed_lr0001.csv'),
    ('Fixed lr=0.001, second exp.)', './logs/w3_logged_metrics_fixed_lr0001_exp2.csv'),
    # ('ResNet20 (Adam, lr=0.05)', './logs/w2_logged_metrics_lr005.csv'),
]

axes, colors = None, None
for i, (label, log_dir) in enumerate(logs_list):

    print("Plotting...\tlabel:{}\tdir:{}".format(label, log_dir))
    LOG_METRICS_DIR = Path(log_dir).resolve()
    logged_metrics = load_logged_metrics(LOG_METRICS_DIR)

    fig, axes, colors = plot_metrics_lab03(
                                    logged_metrics,
                                    dir_fig='logs/w3_logged_metrics_compared.png',
                                    label=label,
                                    axes = axes,
                                    colors = colors,
                                    plotfig=i == (len(logs_list) - 1),
                                    savefig=False)
