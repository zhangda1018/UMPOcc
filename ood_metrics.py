import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def calculate_ood_metrics(out, label, total_count):
    precision, recall, _ = precision_recall_curve(label, out)
    return auc(recall, precision)
