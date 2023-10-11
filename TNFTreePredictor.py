import itertools
import numpy as np
import logging
import networkx as nx
from KPH import KPH, TreePredictor, get_n_kps_from_topic_scores_df
from utils import edges_to_relations


def get_roots(G):
    return [c for c in G.nodes if G.out_degree(c) == 0]


def get_leaves(G):
    return [c for c in G.nodes if G.in_degree(c) == 0]


def get_sum_subtree(G, n, node_to_value, node_to_subtree_value):
    children = list(G.predecessors(n))
    if len(children) == 0:
        node_to_subtree_value[n] = node_to_value[n]
        return node_to_subtree_value[n]
    node_to_subtree_value[n] = node_to_value[n] + np.sum(
        [get_sum_subtree(G, child, node_to_value, node_to_subtree_value) for child in children])
    return node_to_subtree_value[n]


def get_sum_subtrees(G, node_to_value):
    roots = get_roots(G)
    node_to_sum = {}
    for r in roots:
        get_sum_subtree(G, r, node_to_value, node_to_sum)
    return node_to_sum


def get_sum_up_route(G, n, node_to_value, node_to_sum):
    if node_to_sum.get(n):
        return node_to_sum[n]
    parents = list(G.successors(n))  # only one
    assert len(parents) <= 1
    if len(parents) == 0:
        node_to_sum[n] = node_to_value[n]
        return node_to_value[n]
    p = parents[0]
    p_sum = get_sum_up_route(G, p, node_to_value, node_to_sum)

    node_to_sum[n] = p_sum + node_to_value[n]
    return node_to_sum[n]


def get_sum_up_routes(G, node_to_value):
        leaves = get_leaves(G)
        node_to_sum = {}
        for l in leaves:
            get_sum_up_route(G, l, node_to_value, node_to_sum)
        return node_to_sum

def sort_dict_items_by_value_then_key(d, reverse=True):
    d_items = list(d.items())
    d_items.sort(key=lambda x: x[0])
    d_items.sort(key=lambda x: x[1], reverse=reverse)
    return d_items

def tree_to_full_graph(clusters, relations, pair_to_weight):
    G = nx.DiGraph()
    nodes = list(set(itertools.chain(*clusters)))
    edges = []
    for c in clusters:
        edges.extend([(i, j, pair_to_weight[(i, j)]) for i, j in itertools.permutations(c, 2)])

    for e in relations:
        edges.extend([(i, j, pair_to_weight[(i, j)])
                      for i in clusters[e[1]]
                      for j in clusters[e[0]]])

    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([e[0], e[1], {"weight": e[2]}] for e in edges)
    for i in G.nodes:
        for j in nx.descendants(G, i):
            G.add_weighted_edges_from([(i, j, {"weight": pair_to_weight[(i, j)]})])
    return G


class TNFTreePredictor(TreePredictor):
    
    def __init__(self, threshold, pairwise_scores_df, tncf=True):
        super().__init__(threshold, pairwise_scores_df)
        self.topic = list(pairwise_scores_df["topic"])[0]
        self.domain = list(pairwise_scores_df["domain"])[0]
        self.pairwise_scores_df = pairwise_scores_df
        self.n_kps = get_n_kps_from_topic_scores_df(self.pairwise_scores_df)
        pair_to_score = dict(
            zip(zip(list(pairwise_scores_df["i"]), list(pairwise_scores_df["j"])), list(pairwise_scores_df["score"])))
        self.pair_to_weight = {p[0]: p[1] - threshold for p in pair_to_score.items()}
        self.tncf = tncf

        full_graph = self.get_full_graph_from_topic_scores(self.threshold)
        self.reduced_tree = self.get_reduced_graph(full_graph)

    def get_kph(self, clusters, relations):
        return KPH(clusters, relations, n_kps=self.n_kps, topic=self.topic, domain=self.domain)

    def get_graph_score(self, G):
        return np.sum([G.get_edge_data(e[0],e[1])["weight"]["weight"] for e in G.edges])

    def remove_node_from_graph(self, G, v):
        return self.remove_nodes_from_graph(G, [v])

    def remove_nodes_from_graph(self, G, v_list):
        G1 = nx.DiGraph()
        G1.add_nodes_from(list(filter(lambda x:x not in v_list, G.nodes)))
        G1.add_edges_from(list(filter(lambda e: e[0] not in v_list and e[1] not in v_list, G.edges)))
        return G1

    def reduced_tree_to_graph(self, reduced_tree):
        return tree_to_full_graph(reduced_tree['clusters'], reduced_tree['relations'], self.pair_to_weight)

    def get_hierarchy(self, max_iterations = 8):
        if self.n_kps == 1:
            return self.get_kph(clusters = [0], relations = [])

        G = self.reduced_tree_to_graph(self.reduced_tree)

        for i in range(max_iterations):
            start_iter_graph_score = self.get_graph_score(G)
            for v in range(self.n_kps):
                G_without_v = self.remove_node_from_graph(G, v)
                reduced_tree_no_v = self.get_reduced_graph(G_without_v)

                new_reduced_tree, max_score, case = self.find_best_reattachment(reduced_tree_no_v.copy(), v)
                G = self.reduced_tree_to_graph(new_reduced_tree)
                new_graph_score = self.get_graph_score(G)

            if self.tncf and len(new_reduced_tree["clusters"]) > 1:
                multi_clusters = list(filter(lambda x:len(x)>1, new_reduced_tree["clusters"]))
                for c in multi_clusters:
                    if c not in new_reduced_tree["clusters"]: #another cluster was merged to it in the current iteration
                        continue
                    G_without_c = self.remove_nodes_from_graph(G, c)
                    reduced_tree_no_c = self.get_reduced_graph(G_without_c)
                    new_reduced_tree, max_score, case = self.find_best_reattachment_for_cluster(reduced_tree_no_c.copy(),c)
                    G = self.reduced_tree_to_graph(new_reduced_tree)
                    new_graph_score = self.get_graph_score(G)
            if new_graph_score == start_iter_graph_score:
                break

            if i == max_iterations - 1:
                logging.info(f"No convergence in {max_iterations} iteration")
        self.reduced_tree = new_reduced_tree
        return self.get_kph(new_reduced_tree["clusters"],new_reduced_tree["relations"])

    def get_cluster_score(self, cluster_a, cluster_b):
        return np.mean([self.pair_to_weight[(i,j)] for i in cluster_a for j in cluster_b])

    def get_full_graph_from_topic_scores(self, init_thr):
        G = nx.DiGraph()
        scores_df = self.pairwise_scores_df
        edges_pairs_df = scores_df[scores_df["score"] > init_thr]
        edges = list(zip(edges_pairs_df["i"], edges_pairs_df["j"]))
        G.add_nodes_from(list(range(self.n_kps)))
        G.add_edges_from(edges)
        return G

    def choose_one_parent(self, TR, scc_id_to_kp_ids):
        node_to_parent = {n: [] for n in TR.nodes()}
        for e in TR.edges:
            node_to_parent[e[0]].append(e[1])
        node_to_multi_parents = dict(filter(lambda x: len(x[1]) > 1, node_to_parent.items()))
        for n, parents in node_to_multi_parents.items():
            parent_to_cluster_len = {p: len(scc_id_to_kp_ids[p]) for p in parents}
            max_val = np.max(list(parent_to_cluster_len.values()))
            max_size_clusters = dict(filter(lambda x: x[1] == max_val, parent_to_cluster_len.items()))
            if len(max_size_clusters) == 1:
                chosen_p = list(max_size_clusters.keys())[0]
            else:
                max_val_ps = list(max_size_clusters.keys())
                max_clusters_score = -1000000
                chosen_p = None
                for p in max_val_ps:
                    cluster_score = self.get_cluster_score(scc_id_to_kp_ids[n], scc_id_to_kp_ids[p])
                    if cluster_score > max_clusters_score:
                        max_clusters_score = cluster_score
                        chosen_p = p
            for p in parents:
                if p != chosen_p:
                    TR.remove_edge(n, p)
        return TR

    def get_reduced_graph(self, G):
        # collapse ssc
        sccs = list(nx.strongly_connected_components(G))
    
        scc_id_to_kp_ids = {i: scc for i,scc in enumerate(sccs)}
        kp_id_to_scc_id = {}
        for scc_id,scc_kps in scc_id_to_kp_ids.items():
            kp_id_to_scc_id.update({kp_id:scc_id for kp_id in scc_kps})
    
        G2 = nx.DiGraph()
        g2_nodes = scc_id_to_kp_ids.keys()
        g2_edges = set([(kp_id_to_scc_id[e[0]], kp_id_to_scc_id[e[1]]) for e in G.edges])

        g2_edges = list(filter(lambda e: e[0]!=e[1], g2_edges))
        G2.add_nodes_from(g2_nodes)
        G2.add_edges_from(g2_edges)
    
        # transitivity reduction
        TR = nx.transitive_reduction(G2)
    
        # if a NODE HAS TWO PARENTS: 1. CHOOSE THE ONE WITH LARGER CLUSTER 2. TIE BREAKER - larger intra-cluster score
        TR = self.choose_one_parent(TR, scc_id_to_kp_ids)
    
        # convert TR to tree:
        tree = {
            "clusters" : [list(scc_id_to_kp_ids[i]) for i in range(len(scc_id_to_kp_ids))],
            "relations" : edges_to_relations(TR.edges)
        }
        return tree

    def find_best_reattachment_for_cluster(self, reduced_tree, c_kp_list):
        # G: graph of clusters (nodes)
        G = nx.DiGraph()
        clusters_ids = list(range(len(reduced_tree["clusters"])))
        edges = [[e[1], e[0]] for e in reduced_tree["relations"]]
        G.add_nodes_from(clusters_ids)
        G.add_edges_from(edges)

        sum_c_in_weigths_inside, sum_c_out_weigths_inside = {}, {}
        for c in clusters_ids:
            sum_c_in_weigths_inside[c] = np.sum(
                [self.pair_to_weight[(kp_id, c_kp_id)] for kp_id in reduced_tree["clusters"][c] for c_kp_id in c_kp_list])
            sum_c_out_weigths_inside[c] = np.sum(
                [self.pair_to_weight[(c_kp_id, kp_id)] for kp_id in reduced_tree["clusters"][c]  for c_kp_id in c_kp_list])

        c_to_v_in = get_sum_subtrees(G, sum_c_in_weigths_inside)
        c_to_v_out = get_sum_up_routes(G, sum_c_out_weigths_inside)

        # case 1
        add_to_c_scores = {c: c_to_v_in[c] + c_to_v_out[c] for c in clusters_ids}
        max_c_case1, max_c_scores_case1 = sort_dict_items_by_value_then_key(add_to_c_scores)[0]

        # case 2
        c_to_child_to_scores = {c: {d: c_to_v_in[d] for d in G.predecessors(c) if c_to_v_in[d] > 0} for c in clusters_ids}
        c_to_children_score = {c: np.sum(list(c_to_child_to_scores[c].values())) for c in c_to_child_to_scores}
        add_as_c_child_to_score = {c: c_to_v_out[c] + c_to_children_score[c] for c in c_to_child_to_scores}

        max_c_case_2, max_c_scores_case2 = sort_dict_items_by_value_then_key(add_as_c_child_to_score)[0]
        d_list_case2 = list(c_to_child_to_scores[max_c_case_2].keys())

        # case 3
        roots = get_roots(G)
        roots_to_add = {r for r in roots if c_to_v_in[r] > 0}
        add_to_roots_score = np.sum([c_to_v_in[r] for r in roots_to_add])

        max_scores = [max_c_scores_case1, max_c_scores_case2, add_to_roots_score]
        max_score = np.max(max_scores)
        case = 1 + np.argmax(max_scores)
        if max_score == max_c_scores_case1:
            reduced_tree["clusters"][max_c_case1].extend(c_kp_list)
            return reduced_tree, max_score, case

        reduced_tree["clusters"].append(c_kp_list)
        cid = len(reduced_tree["clusters"]) - 1

        if max_score == max_c_scores_case2:
            reduced_tree["relations"].append([max_c_case_2, cid])
            reduced_tree["relations"].extend([[cid, d] for d in d_list_case2])
            for d in d_list_case2:
                if [max_c_case_2, d] in reduced_tree["relations"]:
                    reduced_tree["relations"].remove([max_c_case_2, d])

        else:  # case3
            reduced_tree["relations"].extend([[cid, r] for r in roots_to_add])

        return reduced_tree, max_score, case

    
    def find_best_reattachment(self, reduced_tree, v_node_id):
        # G: graph of clusters (nodes)
        G = nx.DiGraph()
        clusters_ids = list(range(len(reduced_tree["clusters"])))
        edges = [[e[1],e[0]] for e in reduced_tree["relations"]]
        G.add_nodes_from(clusters_ids)
        G.add_edges_from(edges)
    
        sum_v_in_weigths_inside, sum_v_out_weigths_inside = {}, {}
        for c in clusters_ids:
            sum_v_in_weigths_inside[c] = np.sum([self.pair_to_weight[(kp_id, v_node_id)] for kp_id in reduced_tree["clusters"][c]])
            sum_v_out_weigths_inside[c] = np.sum([self.pair_to_weight[(v_node_id, kp_id)] for kp_id in reduced_tree["clusters"][c]])
    
        c_to_v_in = get_sum_subtrees(G, sum_v_in_weigths_inside)
        c_to_v_out = get_sum_up_routes(G, sum_v_out_weigths_inside)
    
        #case 1
        add_to_c_scores = {c: c_to_v_in[c] + c_to_v_out[c] for c in clusters_ids}
        max_c_case1, max_c_scores_case1 = sort_dict_items_by_value_then_key(add_to_c_scores)[0]
    
        # case 2
        c_to_child_to_scores  = {c: {d: c_to_v_in[d] for d in G.predecessors(c) if c_to_v_in[d] > 0} for c in clusters_ids}
        c_to_children_score = {c: np.sum(list(c_to_child_to_scores[c].values())) for c in c_to_child_to_scores}
        add_as_c_child_to_score = {c: c_to_v_out[c] + c_to_children_score[c] for c in c_to_child_to_scores}

        max_c_case_2, max_c_scores_case2 = sort_dict_items_by_value_then_key(add_as_c_child_to_score)[0]
        d_list_case2 = list(c_to_child_to_scores[max_c_case_2].keys())

        # case 3
        roots = get_roots(G)
        roots_to_add = {r for r in roots if c_to_v_in[r] > 0}
        add_to_roots_score = np.sum([c_to_v_in[r] for r in roots_to_add])

        max_scores = [max_c_scores_case1, max_c_scores_case2, add_to_roots_score]
        max_score = np.max(max_scores)
        case = 1+ np.argmax(max_scores)
        if max_score == max_c_scores_case1:
            reduced_tree["clusters"][max_c_case1].append(v_node_id)
            return reduced_tree, max_score, case

        reduced_tree["clusters"].append([v_node_id])
        v_node_cid = len(reduced_tree["clusters"]) - 1

        if max_score == max_c_scores_case2:
            reduced_tree["relations"].append([max_c_case_2, v_node_cid])
            reduced_tree["relations"].extend([[v_node_cid, d] for d in d_list_case2])
            for d in d_list_case2:
                if [max_c_case_2, d] in reduced_tree["relations"]:
                    reduced_tree["relations"].remove([max_c_case_2, d])

        else: # case3
            reduced_tree["relations"].extend([[v_node_cid, r] for r in roots_to_add])

        return reduced_tree, max_score, case


class ReducedForestPredictor(TNFTreePredictor):
    def init(self, threshold, pairwise_scores_df):
        super().__init__(threshold, pairwise_scores_df)

    def get_hierarchy(self):
        return self.get_kph(self.reduced_tree["clusters"], self.reduced_tree["relations"])
