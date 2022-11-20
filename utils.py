from config import cfg
from argparse import Namespace

cfg = Namespace(**cfg)
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def log_roc_curve(fpr, tpr, tensorboard, epoch, tag, auroc):
    for i, label in enumerate(cfg.label_tags):
        fig = plt.figure()
        auroc_i = round(auroc[i].item(), 3)
        plt.plot(
            fpr[i].cpu().numpy(),
            tpr[i].cpu().numpy(),
            label=f"auc = {auroc_i}",
        )
        plt.legend()
        tensorboard.add_figure(
            f"roc_curve_{tag}/{label}",
            fig,
            global_step=epoch,
            walltime=None,
        )
        plt.close(fig)


def log_confusion_matrix(cm, tensorboard, epoch, tag):
    for i, label in enumerate(cfg.label_tags):
        fig = plt.figure()
        df_cm = pd.DataFrame(cm[i].cpu().int().numpy(), range(2), range(2), dtype=int)
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d")
        tensorboard.add_figure(
            f"confusion_matrix_{tag}/{label}",
            fig,
            global_step=epoch,
            walltime=None,
        )
        plt.close(fig)


def log_metrics_table(metrics_dict, tensorboard, epoch, tag):

    metrics_dict = {k: v.cpu().numpy() for k, v in metrics_dict.items()}
    df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=cfg.label_tags)
    fig = plt.figure()
    sn.set(font_scale=1.0)  # for label size
    sn.heatmap(df, annot=True, annot_kws={"size": 12}, fmt=".3f")
    tensorboard.add_figure(
        f"metrics_table_{tag}",
        fig,
        global_step=epoch,
        walltime=None,
    )
    plt.close(fig)
