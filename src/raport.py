# Author: Oskar Bartosz

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def plot_classification_raport(title, acc, prec, rec, cm, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle(title)
    ax.set_title(f'     Accuracy: {acc:.4f}   Precision: {prec:.4f}    Recall: {rec:.4f}', fontsize=8)
    df_cm = pd.DataFrame(cm, index=labels,
                         columns=labels)
    sn.heatmap(df_cm, ax=ax, annot=True, cmap=sn.cm.rocket_r, fmt='g')
    fig.savefig(f"figures/{title.replace(' ', '_')}.png")
    plt.clf()


def make_raport(test_targets, predictions, label="Raport"):
    acc = accuracy_score(test_targets, predictions)
    prec = precision_score(test_targets, predictions, average='macro')
    rec = recall_score(test_targets, predictions, average='macro')
    cm = confusion_matrix(test_targets, predictions)
    print(label)
    print(f'Accuracy: {acc}')
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"Confussion Matrix:\n{cm}")
    plot_classification_raport(label, acc, prec, rec, cm)
    return (acc, prec, rec, cm)
