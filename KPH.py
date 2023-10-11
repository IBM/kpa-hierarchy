import itertools
from collections import Counter
from typing import List

import networkx as nx
import pandas as pd


def get_n_kps_from_topic_scores_df(df_scores):
    return df_scores["i"].max() + 1

class KPH:
    def __init__(self, clusters: List[List[int]], relations: List[List[int]], n_kps: int, topic: str, domain: str):
        """
        :param n_kps: number of kps
        :param clusters: list of ordered clusters, each cluster is a list of the indices of the kps in the clusters
        :param relations: list of relations between clusters. Each relation is (c_id1, c_id2) where c_id1 is the index
         of the more general cluster and *c_id2* is the index of the more specific cluster
        :param domain: PC, Hotel or Restaurant.
        :param topic: an id associated with the business.
        """
        self.clusters = clusters
        self.relations = relations
        self.n_kps = int(n_kps)
        self.domain = domain
        self.topic = topic
        self.assert_valid_kph()

    @classmethod
    def from_scores_df(cls,  clusters: List[List[int]], relations: List[List[int]], df_scores: pd.DataFrame):
        """
        :param clusters: clusters: list of ordered clusters, each cluster is a list of the indices of the kps in the clusters
        :param relations: relations: list of relations between clusters. Each relation is (c_id1, c_id2) where c_id1 is the index
         of the more general cluster and *c_id2* is the index of the more specific cluster
        :param df_scores: the dataframe that contains the kps, domain, topic and kps
        :return:
        """
        n_kps = get_n_kps_from_topic_scores_df(df_scores)
        topic = list(df_scores['topic'])[0]
        domain = list(df_scores['domain'])[0]
        return KPH(clusters, relations, n_kps, topic, domain)

    @classmethod
    def from_dict(cls, topic_dict):
        n_kps = len(topic_dict["kps"])
        topic = topic_dict["topic"]
        domain = topic_dict["domain"]
        clusters = topic_dict["clusters"]
        relations = topic_dict["relations"]
        return KPH(clusters, relations, n_kps, topic, domain)

    def assert_valid_kph(self):
        """
        : raise an exception if the current object is not a valif KPH.
        """
        kps_in_clusters = sorted(list(itertools.chain(*self.clusters)))
        kps_expected = list(range(self.n_kps))
        correct_clustering = kps_in_clusters == kps_expected
        error_msg = ""
        if not correct_clustering:
            error_msg += f"\tKps not clustered correctly: {self.clusters}\n\t\t"
            missing_kps = set(kps_expected).difference(set(kps_in_clusters))
            unexpected_kps = set(kps_in_clusters).difference(set(kps_expected))
            kps_counter = Counter(kps_in_clusters)
            repeating_kps = set(filter(lambda x:kps_counter[x]>1, kps_in_clusters))
            if missing_kps:
                error_msg+= f"missing_kps {missing_kps}, "
            if unexpected_kps:
                error_msg += f"unexpected_kps {unexpected_kps}, "
            if repeating_kps:
                error_msg += f"kps appear in more than one cluster {repeating_kps}, "
            if error_msg:
                error_msg += "\n"

        g = nx.DiGraph()
        g.add_edges_from(self.relations)
        cycles = list(nx.simple_cycles(g))

        n_parents_dict = Counter([r[1] for r in self.relations])
        n_mul_parents_dict = dict(filter(lambda x: x[1] > 1, n_parents_dict.items()))
        nodes_have_a_single_parents = len(n_mul_parents_dict) == 0

        if len(cycles) > 0 or  not nodes_have_a_single_parents:
            error_msg += f"\tInvalid relations between kps:{self.relations}\n"

        if len(cycles) > 0:
            error_msg += f"\t\tCycles found in graph: {str(cycles)}\n"

        if not nodes_have_a_single_parents:
            error_msg += f"\t\tNodes have multiple parents: {list(n_mul_parents_dict.keys())}\n"

        assert len(error_msg) == 0, "Not a valid KPH:\n" + error_msg


    def to_json_line(self):
        return self.__dict__

    def get_viz(self, ordered_kps):
        """
        :return: a string with a user-friendly representation of the tree hierarchy
        """
        node2kp = {i: "; ".join([ordered_kps[x] for x in cluster])
                   for i, cluster in enumerate(self.clusters)}
        graph = nx.DiGraph()
        graph.add_nodes_from(node2kp.keys())
        graph.add_edges_from(self.relations)
        under_root = [node for node in graph.nodes if len(graph.in_edges(node)) == 0]
        viz = "".join([self.get_node_viz(graph, child, 0, node2kp) for child in under_root])
        return viz

    def get_node_viz(self, graph, node, level, node2kp):
        viz = "\t" * level + ("|- " if level != 0 else "") + node2kp[node] + "\n"
        for k in nx.DiGraph.neighbors(graph, node):
            viz += self.get_node_viz(graph, k, level + 1, node2kp)
        return viz


class TreePredictor:
    def __init__(self, threshold: float, pairwise_scores_df: pd.DataFrame):
        """
        :param threshold: the decision threshold for setting a relation between two key points.
        :param pairwise_scores_df: a dataframe with the columns: i (specific kp idx), j (general kp idx), score (pairwise score)
        """
        self.threshold = threshold
        self.pairwise_scores_df = pairwise_scores_df

    def get_hierarchy(self) -> KPH:
        raise NotImplementedError()
