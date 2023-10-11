import pandas as pd
from torch.utils import data
from itertools import permutations, chain
import jsonlines
import networkx as nx


class Pairwise(data.Dataset):
    def __init__(self, data_path, domains=None, topics=[]):
        super(Pairwise, self).__init__()
        self.data_path = data_path

        with jsonlines.open(self.data_path, "r") as f:
            self.topics = {x["topic"]: x for x in f}

        if domains is not None and len(topics) > 0:
            raise ValueError(domains, topics)

        if domains is not None:
            self.topics = {k: v for k, v in self.topics.items() if v["domain"] in domains}

        if len(topics) > 0:
            self.topics = {k: v for k, v in self.topics.items() if k in topics}

        selected_topics = [x["topic"] for x in self.topics.values()]
        self.with_label = "clusters" in self.topics[selected_topics[0]]
        # print(self.topics.keys())

        self.pairs = list(chain.from_iterable([self._get_pairwise_topic(topic)
                                               for _, topic in self.topics.items()]))

    def get_pairs_df(self):
        return pd.DataFrame(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

    def _get_pairwise_topic(self, topic):
        if self.with_label:
            kp2cluster = {}
            for cluster_id, cluster in enumerate(topic["clusters"]):
                for kp in cluster:
                    kp2cluster[kp] = cluster_id

            G = nx.DiGraph()
            G.add_nodes_from(range(len(topic["clusters"])))
            # a directed edge in the graph is an edge between general and specific kps (e.g food quality is subpar --> the pizza is bad)
            G.add_edges_from([(b, a) for a, b in topic["relations"]])

        for a, b in permutations(range(len(topic["kps"])), 2):
            pair = {
                "general": topic["kps"][a],
                "specific": topic["kps"][b],
                "general_idx": a,
                "specific_idx": b,
                "topic": topic["topic"],
                "domain": topic["domain"]
            }
            if self.with_label:
                pair["label"] = 1 if kp2cluster[a] == kp2cluster[b] or nx.has_path(G, kp2cluster[a],
                                                                                   kp2cluster[b]) else 0

            yield pair
