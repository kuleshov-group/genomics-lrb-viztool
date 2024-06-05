"""Variant Effect Prediction Fine-tuned plotting functions.

"""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import average_precision_score, roc_auc_score


def plot_aucroc_auprc(
    df: pd.DataFrame,
    model_col: str = 'model',
    label_col: str = 'labels',
    pred_col: str = 'probability_1'
):
    _, axs = plt.subplots(1, 2, figsize=(20, 8))
    for ax, score_fn, score_str in zip(axs, [roc_auc_score, average_precision_score], ['AUCROC', 'AUPRC']):
        scores = {model_col: [], score_str: []}
        for model_name, group in df.groupby(model_col):
            group = group.dropna(subset=[label_col, pred_col])
            aucroc = score_fn(group[label_col], group[pred_col])
            scores[model_col].append(model_name)
            scores[score_str].append(aucroc)
        scores_df = pd.DataFrame.from_dict(scores)
        # Plot scores
        sns.barplot(
            data=scores_df,
            x=model_col,
            y=score_str,
            hue=model_col,
            dodge=False,
            ax=ax
        )
        ax.tick_params(axis='x', labelrotation=60)
        ax.set_xlabel('')
        ax.set_ylabel(score_str)
        # Display bar values
        # (See: https://stackoverflow.com/questions/55586912/seaborn-catplot-set-values-over-the-bars)
        for c in ax.containers:
            bar_labels = [f'{v.get_height():0.3f}' for v in c]
            ax.bar_label(c, labels=bar_labels, label_type='center', color='white', weight='bold')
    plt.show()

    
def plot_aucroc_auprc_by_bucket(
    df: pd.DataFrame,
    buckets: List[Tuple[Union[int, float]]],
    bucket_col: str = 'tss_dist',
    bucket_display_str: str = 'Distance to TSS',
    model_col: str = 'model',
    label_col: str = 'labels',
    pred_col: str = 'probability_1'
):
    for idx, (score_fn, score_str) in enumerate(zip([roc_auc_score, average_precision_score], ['AUCROC', 'AUPRC'])):
        scores = {model_col: [], bucket_display_str: [], score_str: [], 'n': []}
        for model_name, group in df.groupby(model_col):
            group = group.dropna(subset=[label_col, pred_col])
            for bucket in buckets:
                filtered_group = group[
                    (group[bucket_col] >= bucket[0]) &
                    (group[bucket_col] < bucket[1])
                    ]
                score = score_fn(filtered_group[label_col], filtered_group[pred_col])
                scores[model_col].append(model_name)
                scores[bucket_display_str].append(bucket)
                scores[score_str].append(score)
                scores['n'].append(len(filtered_group[label_col]))
        scores_df = pd.DataFrame.from_dict(scores)
        if idx == 0:
            display(
                scores_df.pivot_table(index=bucket_display_str, columns=model_col, values='n', aggfunc='sum',
                                      fill_value=0)
            )

        # Plot AUC-ROC scores
        g = sns.catplot(
            scores_df,
            x=model_col,
            y=score_str,
            col=bucket_display_str,
            hue=model_col,
            kind='bar',
            dodge=False,
            height=8,
            aspect=1.5,
        )
        g.set(xlabel='', ylabel=score_str)
        g.set_xticklabels(rotation=60, fontsize=16)
        g.set_ylabels(fontsize=16)
        # Display bar values
        # (See: https://stackoverflow.com/questions/55586912/seaborn-catplot-set-values-over-the-bars)
        for ax in g.axes.ravel():
            title = ax.title.get_text()
            ax.set_title(title, fontsize=24)
            for c in ax.containers:
                labels = [f'{v.get_height():0.3f}' for v in c]
                ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=14)
    plt.show()


def plot_aucroc_auprc_by_annotation(
    df: pd.DataFrame,
    annotation_col: str,
    model_col: str = 'model',
    label_col: str = 'labels',
    pred_col: str = 'probability_1'
):
    for idx, (score_fn, score_str) in enumerate(zip([roc_auc_score, average_precision_score], ['AUCROC', 'AUPRC'])):
        scores = {model_col: [], annotation_col: [], score_str: [], 'n': [], 'pos': []}
        for (model_name, annotation_value), group in df.groupby([model_col, annotation_col]):
            group = group.dropna(subset=[label_col, pred_col])
            score = score_fn(group[label_col], group[pred_col])
            scores[model_col].append(model_name)
            scores[annotation_col].append(annotation_value)
            scores[score_str].append(score)
            scores['n'].append(len(group[label_col]))
            scores['pos'].append(group[label_col].sum())
        scores_df = pd.DataFrame.from_dict(scores)
        if idx == 0:
            display(
                scores_df.pivot_table(index=annotation_col, columns=model_col, values=['n', 'pos'], aggfunc='first', fill_value=0)
            )

        # Plot AUC-ROC scores
        g = sns.catplot(
            scores_df,
            x=model_col,
            y=score_str,
            col=annotation_col,
            hue=model_col,
            kind='bar',
            dodge=False,
            height=8,
            aspect=1.5,
        )
        g.set(xlabel='', ylabel=score_str)
        g.set_xticklabels(rotation=60, fontsize=16)
        g.set_ylabels(fontsize=16)
        g.set_titles(template=f"{annotation_col}="+"{col_name}")
        # Display bar values
        # (See: https://stackoverflow.com/questions/55586912/seaborn-catplot-set-values-over-the-bars)
        for ax in g.axes.ravel():
            title = ax.title.get_text()
            ax.set_title(title, fontsize=24)
            for c in ax.containers:
                labels = [f'{v.get_height():0.3f}' for v in c]
                ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=14)
    plt.show()
