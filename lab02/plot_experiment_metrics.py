"""Plot the metrics for the experiment log files"""
from pathlib import Path

from image_classification.data import load_logged_metrics, plot_metrics

logs_list = [
    # ('ResNet20 (Adam, lr=0.001)', './logs/w2_logged_metrics.csv'),
    ('ResNet20 (Adam, lr=0.001)', './logs/w2_logged_metrics_lr0001_augmented.csv'),
    # ('ResNet20 (Adam, lr=0.05)', './logs/w2_logged_metrics_lr005.csv'),
    ('ResNet20 (Adam, lr=0.05)', './logs/w2_logged_metrics_lr005_augmented.csv'),
    # ('ResNet20 (Adam, lr=0.5)', './logs/w2_logged_metrics_lr05.csv'),
]

fig, axes, colors = None, None, None
for i, (label, log_dir) in enumerate(logs_list):

    print("Plotting...\tlabel:{}\tdir:{}".format(label, log_dir))
    LOG_METRICS_DIR = Path(log_dir).resolve()
    logged_metrics = load_logged_metrics(LOG_METRICS_DIR)

    fig, axes, colors = plot_metrics(logged_metrics,
                                    dir_fig = 'logs/w2_logged_metrics_compared.png',
                                    label = label,
                                    fig = fig,
                                    axes = axes,
                                    colors = colors,
                                    plotfig = i == (len(logs_list) - 1),
                                    savefig = i == (len(logs_list) - 1))
