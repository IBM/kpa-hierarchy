import networkx as nx
from itertools import permutations
import numpy as np
import pandas as pd
from utils import load_kph_dict_from_file


class Evaluation:
    def __init__(self, gold_dic, system_dic, topics=None):
        self.gold = gold_dic
        self.system = system_dic

        if topics is None:
            self.gold = {k: v for k, v in self.gold.items() if k in self.system}
            self.system = {k: v for k, v in self.system.items() if k in self.gold}

        else:
            self.gold = {k: v for k, v in self.gold.items() if k in topics}
            self.system = {k: v for k, v in self.system.items() if k in topics}

        assert self.system.keys() == self.gold.keys(), "Gold and system topics are different"

    @classmethod
    def from_path(cls, gold_path, system_path, topics=None):
        gold = load_kph_dict_from_file(gold_path, topics=topics)
        system = load_kph_dict_from_file(system_path, topics=topics)
        return Evaluation(gold, system, topics)

    def evaluate(self, return_raw_scores=False):
        results = []
        for topic_id in self.gold.keys():
            if topic_id not in self.system:
                continue
            results.append(self.evaluate_topic(topic_id))

        scores = {
            "recall": round(np.average([x["recall"] for x in results]), 4),
            "precision": round(np.average([x["precision"] for x in results]), 4),
            "f1": round(np.average([x["f1"] for x in results]), 4),
        }

        if return_raw_scores:
            scores["per_topic"] = results

        return scores

    def get_f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def get_domain(self, r):
        return self.gold[r["topic"]].domain

    def evaluate_micro(self):
        topics_to_run = self.system.keys()
        topic_and_tp_scores = [tuple([t]) + self.get_tp_scores_for_topic(t) for t in topics_to_run]
        topic_and_tp_scores_df = pd.DataFrame(topic_and_tp_scores, columns=["topic", "tp", "tn", "fp", "fn"])

        tp = np.sum(list(topic_and_tp_scores_df["tp"]))
        fp = np.sum(list(topic_and_tp_scores_df["fp"]))
        fn = np.sum(list(topic_and_tp_scores_df["fn"]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = self.get_f1(precision, recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def get_tp_scores_for_topic(self, topic_id):
        if topic_id not in self.gold or topic_id not in self.system:
            raise ValueError(topic_id)

        gold_topic = self.gold[topic_id]
        sys_topic = self.system[topic_id]

        gold_tree = self._get_topic_graph(gold_topic)
        system_tree = self._get_topic_graph(sys_topic)

        m2c_gold = self._get_mention2cluster(gold_topic)
        m2c_system = self._get_mention2cluster(sys_topic)

        tp, tn, fp, fn = 0, 0, 0, 0

        for a, b in permutations(list(range(gold_topic.n_kps)), 2):
            is_gold_link = m2c_gold[a] == m2c_gold[b] or nx.has_path(gold_tree, m2c_gold[a], m2c_gold[b])
            is_sys_link = m2c_system[a] == m2c_system[b] or nx.has_path(system_tree, m2c_system[a], m2c_system[b])

            if is_gold_link and is_sys_link:
                tp += 1
            elif is_gold_link and not is_sys_link:
                fn += 1
            elif not is_gold_link and is_sys_link:
                fp += 1
            else:
                tn += 1

        return tp, tn, fp, fn

    def evaluate_topic(self, topic_id):
        tp, tn, fp, fn = self.get_tp_scores_for_topic(topic_id)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            "topic_id": topic_id,
            "precision": precision,
            "recall": recall,
            "f1": self.get_f1(precision, recall)
        }

    def _get_mention2cluster(self, topic):
        mention2cluster = {}
        for cluster_id, cluster in enumerate(topic.clusters):
            for kp in cluster:
                mention2cluster[kp] = cluster_id

        return mention2cluster

    def _get_topic_graph(self, topic):
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(topic.clusters))))
        G.add_edges_from(topic.relations)
        return G
