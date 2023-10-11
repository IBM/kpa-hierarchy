import argparse
import logging
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

from conf import thresholds, default_scores_df_path
from utils import init_logger
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def plot_pr_curve_from_domains_results(domain_to_method_to_results, domain_to_method_to_auc, out_path, min_rec = 0.1):
    domains = sorted(domain_to_method_to_results.keys())
    fig, axs = plt.subplots(1, len(domains), figsize=(8*len(domains), 8), squeeze=False)
    for d_i, domain in enumerate(domain_to_method_to_results):
        method_to_results = domain_to_method_to_results[domain]
        ax = axs[0, d_i]
        for m_i, method in enumerate(sorted(method_to_results.keys())):
            results = method_to_results[method]
            auc = domain_to_method_to_auc[domain][method]
            ax.plot(results["recall"], results["precision"], linewidth=4, label=method+ " (AUC:%.3f)" % auc)

            ax.set_xlim([min_rec, 1.01])
            ax.set_xticks(np.arange(min_rec, 1.01,0.1))
            ax.set_ylim([0.0, 1.05])
            ax.set_yticks(np.arange(0, 1.01, 0.1))
            ax.legend(loc="best",fontsize=20)
            ax.set_title(domain,fontsize=24)
            ax.set_xlabel("Recall",fontsize=24)
            if d_i == 0:
                ax.set_ylabel("Precision",fontsize=20)

    out_file = out_path.replace(".csv",".png")
    logging.info(f"Saving graph results to {out_file}")
    plt.savefig(out_file)


def get_max_f1_res(per_thr_scores_df):
    f1_max_idx = np.argmax(list(per_thr_scores_df["f1"]))
    max_f1_dict = dict(per_thr_scores_df.iloc[f1_max_idx])
    return max_f1_dict


def get_global_scores(per_thr_scores_df, min_recall):
    max_f1_dict = get_max_f1_res(per_thr_scores_df)
    df = per_thr_scores_df[per_thr_scores_df["recall"] >= min_recall]
    df = df.sort_values(by=["recall", "precision"], ascending=[True, False]).drop_duplicates(subset=["recall"])
    auc =  np.trapz(y=list(df['precision']), x=list(df['recall']))
    return {'best_thr': max_f1_dict['threshold'], "max_f1": max_f1_dict['f1'], "auc": auc}


def load_pairwise_scores(scores_path, domains = None, topics = None):
    scores_df = pd.read_csv(scores_path)
    scores_df = scores_df.rename(columns={"specific_idx": "i", "general_idx": "j"})
    if domains:
        scores_df = scores_df[scores_df["domain"].isin(domains)]
    if topics:
        scores_df = scores_df[scores_df["topic"].isin(topics)]
    return scores_df


def get_per_thr_results_for_method(df, thresholds, method):
    method_rows = []
    for thr in thresholds:
        scores = np.where(df[method] >= thr, 1, 0)

        p, r, f, _ = precision_recall_fscore_support(df["label"], scores, average="binary", zero_division=0)
        method_rows.append([thr, p, r, f])
    method_df = pd.DataFrame(method_rows, columns=["threshold", "precision", "recall", "f1"])
    return method_df


def get_best_f1_and_thr(scores_df, thresholds, method):
    per_thr_results = get_per_thr_results_for_method(scores_df, thresholds, method)
    max_f1_scores = get_max_f1_res(per_thr_results)
    return max_f1_scores["f1"], max_f1_scores["threshold"]


def get_best_f1_lomo(scores_df, thresholds, method):
    labels = []
    preds = []
    for topic, topic_scores in scores_df.groupby("topic"):
        other_topics_df = scores_df[scores_df["topic"] != topic]
        _, thr = get_best_f1_and_thr(other_topics_df, thresholds, method)
        labels.extend(topic_scores["label"])
        preds.extend([int(s >= thr) for s in topic_scores[method]])

    _, _, best_f_lomo, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return best_f_lomo


def get_best_f1_thr_loo_and_oracle(scores_df, thresholds, method):
    best_f_lomo = get_best_f1_lomo(scores_df, thresholds, method)
    best_f1_oracle, max_thr_oracle = get_best_f1_and_thr(scores_df, thresholds, method)
    return best_f_lomo, best_f1_oracle, max_thr_oracle


def eval_pairwise_scores(scores_path, methods, out_path, domains=None, min_recall = 0.1):
    logging.info(f"Evaluating pairwise scores for scores in :{scores_path}\n\t\t\tmethods:{methods}\n\t\t\toutput_path:{out_path}")
    scores_df = load_pairwise_scores(scores_path, domains)
    domain_to_method_to_results_df = defaultdict(dict)
    domain_to_method_to_auc = defaultdict(dict)
    global_results_rows = []
    if not domains:
        domains = list(set(scores_df["domain"]))

    for domain in domains:
        domain_scores_df = scores_df[scores_df["domain"] == domain]
        for method in methods:

            per_thr_results = get_per_thr_results_for_method(domain_scores_df, thresholds, method)
            global_scores = get_global_scores(per_thr_results, min_recall=min_recall)
            per_thr_results = per_thr_results.sort_values(by=["recall", "precision"], ascending=[True, False]).drop_duplicates(subset=["recall"])
            per_thr_results = per_thr_results[per_thr_results["recall"] >= min_recall]
            domain_to_method_to_results_df[domain][method] = per_thr_results

            domain_to_method_to_auc[domain][method] = global_scores["auc"]

            best_f1_lomo = get_best_f1_lomo(domain_scores_df, thresholds, method)
            global_results_rows.append([domain, method, best_f1_lomo, global_scores["auc"]])

    plot_pr_curve_from_domains_results(domain_to_method_to_results_df, domain_to_method_to_auc, out_path=out_path)

    global_scores_df = pd.DataFrame(global_results_rows, columns = ["domain", "method", 'best_f1_lomo', 'auc'])
    for method, group in global_scores_df.groupby("method"):
        avg_row = ["Avg.", method, np.mean(group["best_f1_lomo"]), np.mean(group["auc"])]
        global_scores_df.loc[len(global_scores_df)] = avg_row
    global_scores_df = global_scores_df.sort_values(by="method")

    global_scores_df.style.set_properties(**{'text-align': 'left'})
    print(tabulate(global_scores_df, showindex=False, headers=global_scores_df.columns, floatfmt=".3f", tablefmt="rst"))
    global_scores_df.to_csv(out_path.replace(".png",".csv"))


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Path to .png file to save the graph", required=True)
    parser.add_argument("--pairwise_scores_file", type=str, help="path to csv with pairwise scores", default=default_scores_df_path)
    parser.add_argument("--methods", nargs='+', help="scoring methods to eval, must be columns in pairwise_scores_df", default=["NLI_BinInc_WL"])
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    eval_pairwise_scores(scores_path=args.pairwise_scores_file, methods=args.methods, out_path=args.output_path)