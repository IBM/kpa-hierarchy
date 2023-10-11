import itertools
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx
from itertools import permutations
from KPH import KPH, TreePredictor, get_n_kps_from_topic_scores_df
from utils import edges_to_relations


class GreedyTreePredictor(TreePredictor):
    def __init__(self, pairwise_scores_df, threshold=0.5):
        super().__init__(threshold, pairwise_scores_df)

    def _get_equivalent_kps(self):
        clusters = self.get_clusters_agg_clustering()
        return clusters

    def get_cluster_score(self, cluster_a, cluster_b):
        return self.pairwise_scores_df[(self.pairwise_scores_df["i"].isin(cluster_a))
                         & (self.pairwise_scores_df["j"].isin(cluster_b))]["score"].mean()

    def _get_hierarchical_relations(self, clusters):
        cluster_ids = np.array(range(len(clusters)))
        cluster_pairs = np.array(list(permutations(cluster_ids, 2)))
        hierarchical_scores = np.array([
            self.get_cluster_score(clusters[a], clusters[b])
            for a, b in cluster_pairs])

        sorted_indices = hierarchical_scores.argsort()[::-1]
        graph = nx.DiGraph(directed=True)
        graph.add_nodes_from(cluster_ids)
        cluster_pairs = cluster_pairs[sorted_indices]
        hierarchical_scores = hierarchical_scores[sorted_indices]

        graph = self.add_edges_to_graph(graph, cluster_pairs, hierarchical_scores)
        return edges_to_relations(graph.edges)

    def add_edges_to_graph(self, graph, cluster_pairs, hierarchical_scores):
        raise NotImplementedError

    def get_clusters_agg_clustering(self):
        df_scores = self.pairwise_scores_df
        n_kps = get_n_kps_from_topic_scores_df(self.pairwise_scores_df)
        dist_mat = np.zeros([n_kps, n_kps])
        pair_to_score = dict(zip(zip(df_scores["i"], df_scores["j"]), df_scores["score"]))
        pair_to_dist = {p: 1 - pair_to_score[p] for p in pair_to_score}
        distance_threshold = 1 - self.threshold
        for i, j in itertools.combinations(range(n_kps), 2):
            dist = np.max([pair_to_dist[(i, j)], pair_to_dist[(j, i)]])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
        clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average",
                                             distance_threshold=distance_threshold)
        labels = clustering.fit(dist_mat).labels_
        n_clusters = len(set(labels))
        clusters = [[] for i in range(n_clusters)]
        for i, l in enumerate(labels):
            clusters[l].append(i)
        return clusters

    def get_hierarchy(self):
        scores = list(self.pairwise_scores_df["score"])
        kp_number = get_n_kps_from_topic_scores_df(self.pairwise_scores_df)

        # trivial cases
        if np.min(scores) > self.threshold:
            clusters = [list(range(kp_number))]
            relations = []

        elif np.max(scores) < self.threshold:
            clusters = [[x] for x in range(kp_number)]
            relations = []

        else:
            clusters = self._get_equivalent_kps()
            relations = self._get_hierarchical_relations(clusters)

        return KPH.from_scores_df(clusters=clusters, relations=relations, df_scores=self.pairwise_scores_df)


class GreedyTreePredictorLocalScore(GreedyTreePredictor):
    def __init__(self, pairwise_scores_df, threshold=0.5):
        super().__init__(threshold, pairwise_scores_df)

    def add_edges_to_graph(self, graph, cluster_pairs, hierarchical_scores):
        for pair, score in zip(cluster_pairs, hierarchical_scores):
            if score < self.threshold:
                break
            child, parent = pair
            # to prevent multiple parents to the same node
            if graph.out_degree(child) == 0:
                # to prevent cycles
                if child not in nx.descendants(graph, parent):
                    graph.add_edge(child, parent)
        return graph


class GreedyTreePredictorBestEdge(GreedyTreePredictor):
    def __init__(self, pairwise_scores_df, threshold=0.5):
        super().__init__(threshold, pairwise_scores_df)

    def add_edges_to_graph(self, graph, cluster_pairs, hierarchical_scores):
        pair_to_score = {tuple(cluster_pairs[i]): hierarchical_scores[i] - self.threshold for i in
                         range(cluster_pairs.shape[0])}
        potential_edges = list(filter(lambda p: pair_to_score[p] >= 0, pair_to_score.keys()))
        graph_score = 0
        while len(potential_edges) > 0:
            max_score = graph_score
            best_e = None
            for e in potential_edges:
                child, parent = e
                if nx.has_path(graph, parent, child) or graph.out_degree(child) > 0:
                    potential_edges.remove(e)  # we're only adding edges, so this edge will always be bad
                else:
                    graph.add_edges_from([e])
                    g_e_score = np.sum([pair_to_score[i, j] for i in graph.nodes for j in nx.descendants(graph, i)])
                    if g_e_score >= max_score:
                        max_score = g_e_score
                        best_e = e
                    graph.remove_edges_from([e])
            if best_e:
                graph.add_edges_from([best_e])
                potential_edges.remove(best_e)
                graph_score = max_score
            else:  # no edge improves graph score
                potential_edges = []
        return graph
