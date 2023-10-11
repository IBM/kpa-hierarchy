import copy
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from tabulate import tabulate
import jsonlines
import argparse
from tqdm import tqdm
from KPH import KPH
from evaluate import Evaluation
import logging
from conf import thresholds, thinkp_path
from eval_pairwise_scores import get_best_f1_thr_loo_and_oracle, load_pairwise_scores
from predict_kph import kph_method_to_predictor_class
from utils import init_logger, filter_topics_by_domains, load_kph_dict_from_file, get_kps_from_pairwise_df


def write_new_trees_for_topics(topics_to_run, tree_method, local_method, pairwise_scores_df, kph_output_path, computed_kphs={}):
    topic_to_thr_to_kph = defaultdict(lambda: {})
    topic_to_df = pairwise_scores_df.groupby("topic")
    trees_built = 0
    total_trees = len(topics_to_run) * len(thresholds)
    for topic_id in tqdm(topics_to_run):
        topic_pairwise_scores_df = topic_to_df.get_group(topic_id)
        topic_scores = list(topic_pairwise_scores_df["score"])
        new_trees = []
        for i, thr in enumerate(thresholds):
            tree_key = (tree_method, local_method, topic_id, thr)
            if tree_key in computed_kphs:
                tree = computed_kphs[tree_key]
                kph = KPH.from_scores_df(tree["clusters"], tree["relations"], topic_pairwise_scores_df)
            else:
                if i == 0 or len(list(filter(lambda x: thresholds[i-1]<x<=thr, topic_scores))) > 0:
                    tree_generator = kph_method_to_predictor_class[tree_method](thr, topic_pairwise_scores_df)
                    kph = tree_generator.get_hierarchy()

                    topic_to_thr_to_kph[topic_id][thr] = kph
                    trees_built += 1
                else: # no scores in threshold bin
                    kph = copy.deepcopy(kph)
                new_trees.append({"tree_method": tree_method,
                                            "pairwise_method": local_method,
                                            "topic":topic_id,
                                            "threshold":thr,
                                            "clusters":kph.clusters,
                                            "relations":kph.relations,
                                            "domain":kph.domain,
                                            "n_kps":kph.n_kps
                                            })

        with jsonlines.open(kph_output_path, "a") as f:
            f.write_all(new_trees)
    logging.info(f"Constructed {trees_built} out of {total_trees}")
    return topic_to_thr_to_kph


def write_trees_for_topics_for_methods(tree_methods, local_methods, pairwise_scores_df, kph_output_path, precomputed_kphs={}):
    logging.info(f"Generating kphs for pairwise methods:{local_methods}, kphs methods:{tree_methods}")

    total_n_methods = len(local_methods) * len(tree_methods)
    i = 0
    topics_to_run = list(set(pairwise_scores_df['topic']))
    for local_method in local_methods:
        pairwise_scores_df_method = pairwise_scores_df.rename(columns={local_method: "score"})
        for tree_method in tree_methods:
            i += 1
            logging.info(f"{i}/{total_n_methods} Generating kphs for pairwise method {local_method}, kph method {tree_method}")
            write_new_trees_for_topics(topics_to_run, tree_method, local_method,
                                        pairwise_scores_df_method, kph_output_path, precomputed_kphs)
            #for topic in topic_to_thr_to_kph:
             #   tree_key = (tree_method, local_method, topic)
              #  predicted_trees[tree_key] = topic_to_thr_to_kph[topic]
    #return predicted_trees


# gold: topic -> kph
# generated_trees_dicts: topic -> {thr > kph}
def eval_get_best_results_from_kph(gold_trees_dict, generated_trees_dict, topics_ids = None):

    if topics_ids is None:
        topics_ids = gold_trees_dict.keys()

    all_results = []
    for thr in thresholds:
        thr_generated_trees = {topic_id : generated_trees_dict[topic_id][thr] for topic_id in topics_ids}
        results = Evaluation(gold_trees_dict, thr_generated_trees).evaluate_micro()
        results["threshold"] = thr
        all_results.append(results)

    best_results = sorted(all_results, key=lambda k:(k["f1"],k["threshold"]), reverse=True)[0]
    return best_results


def get_methods_to_topic_to_thr_to_tree_for_methods(saved_trees_path, local_methods, tree_methods, topics):
    method_pairs = [(tree_method, local_method) for tree_method in tree_methods for local_method in local_methods]
    methods_to_topic_to_thr_to_tree = {method_pair : {topic : {} for topic in topics} for method_pair in method_pairs}

    with jsonlines.open(saved_trees_path, "r") as f:
        saved_trees = [x for x in f]
    for x in saved_trees:
        methods_to_topic_to_thr_to_tree[(x["tree_method"], x["pairwise_method"])][x["topic"]][x["threshold"]] = \
                                        KPH(x["clusters"], x["relations"], n_kps=x["n_kps"], domain=x["domain"], topic = x["topic"])
    return methods_to_topic_to_thr_to_tree


def get_best_f1_loo(saved_trees_path, local_methods, tree_methods, pairwise_scores_df, gold_path, output_dir, domains, save_selected_trees = True):
    logging.info("Computing best F1 loo")
    if not domains:
        domains = list(set(pairwise_scores_df["domain"]))

    topic_to_kps = {t: get_kps_from_pairwise_df(group) for t, group in pairwise_scores_df.groupby("topic")}
    methods_to_topic_to_thr_to_tree = get_methods_to_topic_to_thr_to_tree_for_methods(saved_trees_path, local_methods, tree_methods, list(topic_to_kps.keys()))

    gold_data = load_kph_dict_from_file(gold_path, domains)
    per_domain_results_rows = []
    for local_method in local_methods:
        logging.info(f"local method: {local_method}")
        for d in domains:
            domain_scores_df = pairwise_scores_df[pairwise_scores_df["domain"] == d]
            best_f_lomo, best_f1_oracle, max_thr_oracle = get_best_f1_thr_loo_and_oracle(domain_scores_df, thresholds, local_method)
            per_domain_results_rows.append([d, local_method, "-", best_f_lomo, best_f1_oracle])

        for tree_method in tree_methods:
            logging.info(f"kph method: {tree_method}")
            topic_to_thr_to_kph = methods_to_topic_to_thr_to_tree[(tree_method, local_method)]
            best_trees_per_topic_loo = {}
            best_thresholds_per_topic_loo = {}
            for d in domains:
                gold_data_domain = filter_topics_by_domains(gold_data, [d])
                domain_topic_ids = gold_data_domain.keys()
                t_to_rest_topic_in_d = {t: [t2 for t2 in gold_data_domain.keys() if t2 != t] for t in domain_topic_ids}

                oracle_results = eval_get_best_results_from_kph(gold_data_domain, topic_to_thr_to_kph, topics_ids=domain_topic_ids)
                f1_d_oracle = oracle_results["f1"]
                best_trees_per_topic_loo_d = {}
                for t in tqdm(domain_topic_ids, desc=f"doamin: {d}"):
                    rest_topics_in_domain = t_to_rest_topic_in_d[t]
                    t_best_results_loo = eval_get_best_results_from_kph(gold_data_domain, topic_to_thr_to_kph, topics_ids=rest_topics_in_domain)
                    t_best_thr_loo = t_best_results_loo["threshold"]
                    best_kph_loo = topic_to_thr_to_kph[t][t_best_thr_loo]
                    best_trees_per_topic_loo_d[t] = copy.deepcopy(best_kph_loo)
                    best_thresholds_per_topic_loo[t] = t_best_thr_loo

                f1_d_loo = Evaluation(gold_data_domain, best_trees_per_topic_loo_d).evaluate_micro()["f1"]
                per_domain_results_rows.append([d, local_method, tree_method, f1_d_loo, f1_d_oracle])
                best_trees_per_topic_loo.update(best_trees_per_topic_loo_d)

            if save_selected_trees:
                tree_dir = os.path.join(output_dir, "generated_trees", local_method, tree_method)
                os.makedirs(tree_dir, exist_ok=True)
                trees_dict_file = os.path.join(tree_dir,"generated_trees.jsonl")
                final_tree_dicts = []
                for t, tree in best_trees_per_topic_loo.items():
                    tree_dict = tree.to_json_line()
                    tree_dict["thr"] = best_thresholds_per_topic_loo[t]
                    tree_dict["kps"] = topic_to_kps[t]
                    final_tree_dicts.append(tree_dict)
                    tree_viz_file = os.path.join(tree_dir, f"{t}.txt")
                    with open(tree_viz_file, "w") as f:
                        f.write(tree.get_viz(ordered_kps=topic_to_kps[t]))
                with jsonlines.open(trees_dict_file, "w") as f:
                    f.write_all(final_tree_dicts)

    results_df = pd.DataFrame(per_domain_results_rows, columns=["domain","pairwise_method","tree_method","f1_loo","f1_oracle"])
    for (tree_method, local_method), group in results_df.groupby(["tree_method","pairwise_method"]):
        avg_row = ["Avg.", local_method, tree_method, np.mean(group["f1_loo"]), np.mean(group["f1_oracle"])]
        results_df.loc[len(results_df)] = avg_row
    results_df = results_df.sort_values(by=["domain","pairwise_method","tree_method"])
    logging.info("Printing results:")
    print(tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".3f"))
    out_file = os.path.join(output_dir, "f1_results.csv")
    logging.info(f"Saving results to : {out_file}")
    results_df.to_csv()
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gold_path", type=str, default=thinkp_path)
    parser.add_argument("--pairwise_scores_file", type=str, help="path to pairwise scores csv", default="./eval/pairwise_scores/all_pairwise_scores.csv")
    parser.add_argument("--tree_methods", type=str, nargs='+', default=["tncf"])
    parser.add_argument("--pairwise_methods", type=str, nargs='+', default=["NLI_BinInc_WL"])
    parser.add_argument("--domains", type=str, nargs='+', default=None)
    args = parser.parse_args()
    init_logger()

    kphs_output_path = os.path.join(args.output_dir, "all_generated_kphs.jsonl")
    if os.path.exists(kphs_output_path):
         with jsonlines.open(kphs_output_path, "r") as f:
            cached_kphs_dict = {(x["tree_method"],x["pairwise_method"], x["topic"],x["threshold"]):x for x in f}
    else:
        cached_kphs_dict = {}

    if not os.path.exists(args.output_dir):
         os.makedirs(args.output_dir, exist_ok=True)

    all_pairwise_df = load_pairwise_scores(args.pairwise_scores_file, args.domains)
    write_trees_for_topics_for_methods(args.tree_methods, args.pairwise_methods, all_pairwise_df, kphs_output_path, cached_kphs_dict)
    get_best_f1_loo(kphs_output_path, args.pairwise_methods, args.tree_methods, all_pairwise_df, args.gold_path, args.output_dir, args.domains)


