# ThinkP: A benchmark dataset of key point hierarchies 

## Data Description:

Each Key Point Hierarchy (KPH) is a hierarchical summary of reviews for a business or product in three domains: Hotels, Restaurants and PC.
A KPH is a directed forest, a Directed Acyclic Graph (DAG) where each node has no more than one parent. The nodes are clusters of key points that convey similar ideas, and the edges represent hierarchical relations between clusters. The key points were extracted from the business/product reviews by applying Key Point Analysis.


The ThinkP dataset is described in the following paper:  
[From Key Points to Key Point Hierarchy: Structured and Expressive Opinion Summarization.](https://arxiv.org/abs/2306.03853) Arie Cattan, Lilach Eden, Yoav Kantor and Roy Bar-Haim. ACL 2023.

## License
This dataset is released under [Community Data License Agreement – Sharing, Version 1.0](https://cdla.dev/sharing-1-0/)

## Data files:
The viz subfolder displays the KPH of each business or product in an easy-to-read text file.

The data is available in the file **ThinkP.jsonl**.  
Each line stores a single *KPH*:
- `viz_file_name` : the name of the corresponding file in the viz subfolder.
- `kps` : list of the key points in the KPH.
- `clusters`: list of clusters of equivalent key points. Each cluster is a list storing the indices of the key points in that cluster.
- `relations`: list of relations between clusters. Each relation is *(c_id1, c_id2)* where *c_id1* is the index of the more general cluster and *c_id2* is the index of the more specific cluster, whose key points provide elaboration and support to the key points in *c_id1*.
- `domain`: PC, Hotel or Restaurant.
- `topic`: an id associated with the business.
