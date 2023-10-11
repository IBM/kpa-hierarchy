# kpa-hierarchy-promo

## Scope
This repository contains:

 (1) The ThinkP dataset: a high quality benchmark dataset of key point hierarchies for business and product reviews.  
 (2) NEW: Code for KPH construction and evaluation.

## Data
Information about using the data can be found [here](ThinkP/README.md).


## Setup
In order to run this repo, you need a Python Anaconda environment with all the requirements installed:
```bash 
conda create --name kpa_hierarchy python=3.9
conda activate kpa_hierarchy 
pip install -r requirements.txt
```

## Creating Pairwise Scores:

```bash 
python create_pairwise_scores.py --output_path "./out/pairwise_scores.csv"
```

To generate pairwise scores for the ThinkP dataset, use the `create_pairwise_scores.py` script.
This will load the ThinkP dataset and convert it to a dataframe with the predicted scores for all the key points pair in
each topic. Replace lines 17-19 with your code for computing
the pairwise scores: add a column for the dataframe with a unique method name and the scores for each row.
The scores for the methods reported in the paper are in `eval/pairwise_scores/all_pairwise_scores.csv`.

Arguments:
- `gold_path` : path to the gold data jsonl, default to the path of ThinkP.
- `output_path`: path of csv to save the pairwise scores dataframe. 


## Evaluating Pairwise scores:
```bash
python eval_pairwise_scores.py --output_path "./out/eval_pairwise.csv" --methods APInc NLI BinInc KPA-Match NLI_BinInc_WL
```

This script runs evaluation over the scores computed in the previous section, and outputs:
1. the Precision-Recall graphs per domain.
2. The auc (for recall > 0.1) and best f1 score (using leave-one-topic-out) for choosing the classification threshold for each domain.

Arguments:
- `output_path`: path to the output .png file to be saved. The table with the scores will be saved to a
csv file in the same path.
- `pairwise_scores_file`: path to csv with pairwise scores (defaults to our provided pairwise scores).
- `methods`: list of space seperated methods to evaluate, i.e. columns in the dataframe in pairwise_scores_file


## Constructing KPH

```bash
python predict_kph.py --topic "AV6weBrZFFBfRGCbcRGO4g_neg" --viz --output_dir "./out/build_kph/" --pairwise_method "NLI_BinInc_WL" --tree_method "tncf"
```

Tree construction is performed in two steps: first, computing the pairwise scores for each pair of key points,
and then using the pairwise scores to construct the hierarchy. The first step is done in the previous section,
resulting in the pairwise scores dataframe. To run kph construction from the pairwise scores, run the `predict_kph.py`
script. This script constructs a single KPH, for a given classification threshold, and prints its evaluation measures
against the gold data.

Arguments:
- `gold_path`: path to the gold data jsonl (default to the path of ThinkP).
- `pairwise_scores_file`: path to a csv file with pairwise scores (defaults to our provided pairwise scores).
- `pairwise_methods`: the pairwise methods to use, i.e. a column in the dataframe in pairwise_scores_file (default to NLI_BinInc_WL)
- `threshold`: the decision threshold for counting two kps as related (default 0.5)
- `topic`: the (string) topic id of the business or product to build the kph for
- `output_dir`: path to output directory to save a directory with the jsonl file of the hierarchy and .txt for visualization")
- `viz`: create or not a user friendly visualization of the generated tree
- `tree_methods`: which kph method to use for tree construction, must be a key in `kph_method_to_predictor_class`, the dictionary
in `predict_kph.py`. The construction methods available in the paper are available. 


## Adding a new hierarchy construction method
KPH construction is done using a class that extends `TreePredictor`: its constructor receives a decision threshold and a 
dataframe which contains all the rows for a certain topic in the pairwise scores df, with the relevant pairwise scores column
named "score". The class has a method called `get_hierarchy` that returns a `KPH` object. Both `TreePredictor` and `KPH`
are documented in `KPH.py`. 
Once the class is ready, add an entry to `kph_method_to_predictor_class` with a unique name as key and the class name as value, and run
`predict_kph.py` as explained in the previous section.


## Evaluating KPH constructions

```bash
python eval_kph.py --output_dir ./out/eval_kph --tree_methods reduced_tree greedy_local_score greedy_best_edge tncf --pairwise_methods NLI_BinInc_WL
```

This script first creates and saves all KPHs with all thresholds for all the combinations of the construction methods and pairwise methods.
Then it performs the evaluation, computes the best f1_score (using leave-one-topic-out in each domain) and saves a visualization of the best tree for each combination of methods and topic.

Arguments:
- `output_dir`: required, directory to save trees and evaluation results. previous evaluations in the same dir will be overriden.
the generated trees are saved during the run, so if the execution was terminated or if you want to add more methods to the evaluation,
You can use the same output dir and continue from where you left off
- `gold_path`: path to the gold data jsonl (default to the path of ThinkP).
- `pairwise_scores_file`: path to csv with pairwise scores (defaults to our provided pairwise scores).
- `tree_methods`: list of space seperated KPH construction methods to evaluate, must be keys in `kph_method_to_predictor_class` (as explained in the previous section)
- `pairwise_methods`: list of space seperated pairwise methods to evaluate, i.e. columns in the dataframe in pairwise_scores_file
- `domains`: list of domains to evaluates (by default, run for all domains).

## Citing 
If you are using ThinkP in a publication, please cite the following paper: 

[From Key Points to Key Point Hierarchy: Structured and Expressive Opinion Summarization]()  
Arie Cattan*, Lilach Eden*, Yoav Kantor and Roy Bar-Haim.  
ACL 2023.

## Changelog
Major changes are documented [here](Changelog.md)
