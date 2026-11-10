import json
import os
import datetime
import numpy as np
STATS_PATH = 'statistics.json'
POSITIVE = 1

def _div(a,b):
    return float(a) / float(b) if b else 0.0

def average_precision(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1-y)
    precision = tp / (tp+fp)
    recall = tp / n_pos
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev_recall)*precision))

def compute_metrics(labels, probs, class_names, positive=POSITIVE):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1

    tp = cm[positive, positive]
    fp = cm[:, positive].sum() - tp
    fn = cm[positive,:].sum() - tp
    precision = _div(tp, tp+fp)
    recall = _div(tp, tp+fn)
    f1 = _div(2 * precision * recall, precision + recall)
    per_class_ap = [average_precision(labels == i, probs[:, i]) for i in range(n_classes)]
    return{
        'accuracy': _div(np.trace(cm), cm.sum()),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_ap': per_class_ap,
        'mAP': float(np.mean(per_class_ap)),
        'num_samples': int(cm.sum()),
        'class_names': list(class_names),
    }
def save_statistics(stats, split, path=STATS_PATH):
    out = dict(stats, split=split, timestamp=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    return out
 
 
def load_statistics(path=STATS_PATH):
    """Cached stats from the last evaluation, or None."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
 
 
if __name__ == '__main__':
    # quick self-check: 4 samples, 2 classes
    labels = [0, 0, 1, 1]
    probs = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]
    print(compute_metrics(labels, probs, ['NORMAL', 'PNEUMONIA']))