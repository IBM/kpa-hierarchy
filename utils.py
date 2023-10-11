from logging import getLogger, getLevelName, Formatter, StreamHandler

import jsonlines

from KPH import KPH


def init_logger():
    log = getLogger()
    log.setLevel(getLevelName('INFO'))
    log_formatter = Formatter("%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s [%(threadName)s] ")

    console_handler = StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.handlers = []
    log.addHandler(console_handler)


# In kph relations: [general -> specific ]
# In graph edges: [specific -> general]
def edges_to_relations(edges):
    return [[int(b), int(a)] for (a,b) in edges]


def filter_topics_by_domains(topic_dict, domains=None):
    if not domains:
        return topic_dict
    else:
        topics = list(filter(lambda x: topic_dict[x].domain in domains, topic_dict.keys()))
        return {t: topic_dict[t] for t in topics}


def load_kph_dict_from_file(data_path, domains = None, topics = None):
    with jsonlines.open(data_path, "r") as f:
        kph_dict = {x["topic"]: KPH.from_dict(x) for x in f}
    if domains:
        kph_dict = filter_topics_by_domains(kph_dict, domains)
    if topics:
        kph_dict = dict(filter(lambda x:x[0] in topics, kph_dict.items()))
    return kph_dict


def get_kps_from_pairwise_df(df):
    return list(df.sort_values(by="i").drop_duplicates(subset=["i"])["specific"])
