from pathlib import Path

from image_classification.data import load_logged_metrics, plot_metrics

LOG_METRICS_DIR = Path('./logs/w2_logged_metrics.csv').resolve()

logged_metrics = load_logged_metrics(LOG_METRICS_DIR)

_, _, _ = plot_metrics(logged_metrics,
                        dir_fig='logs/w2_logged_metrics.png',
                        label='ResNet20 (Adam, lr=0.001)',
                        plotfig=True,
                        savefig=True)
