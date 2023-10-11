import argparse
import logging
import os
import jsonlines
from GreedyTreePredictor import GreedyTreePredictorLocalScore, GreedyTreePredictorBestEdge
from TNFTreePredictor import ReducedForestPredictor, TNFTreePredictor
from conf import default_scores_df_path, thinkp_path
from eval_pairwise_scores import load_pairwise_scores
from evaluate import Evaluation
from utils import init_logger, load_kph_dict_from_file, get_kps_from_pairwise_df

kph_method_to_predictor_class = {
    "greedy_local_score": GreedyTreePredictorLocalScore,
    "greedy_best_edge": GreedyTreePredictorBestEdge,
    "reduced_tree": ReducedForestPredictor,
    "tncf": TNFTreePredictor,
    #"your_custum_predictor_name": you_custom_predictor_class
}

if __name__ == "__main__":
  init_logger()
  parser = argparse.ArgumentParser()
  parser.add_argument("--gold_path", type=str, default=thinkp_path)
  parser.add_argument("--pairwise_scores_file", type=str, help="path to pairwise scores jsonl file", default=default_scores_df_path)
  parser.add_argument("--pairwise_method", type=str, help="pairwise scoring method (column in pairwise_scores df)", default="NLI_BinInc_WL")
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument("--topic", type=str, help="topic to build the kph for", required=True)
  parser.add_argument("--output_dir", type=str, required=True, help="path to output directory to save a directory with the jsonl file of the hierarchy and .txt for vizualization")
  parser.add_argument("--viz", action="store_true", help="create or not a user friendly visualization of the generated tree")
  parser.add_argument("--tree_method", type=str, help="which kph method to use for tree construction, should be a key in kph_method_to_predictor_class", default='tncf')
  args = parser.parse_args()

  logging.info(f"Running for topic {args.topic}, pairwise_method {args.pairwise_method}, kph method {args.tree_method}, threshold {args.threshold}")
  pairwise_scores_df = load_pairwise_scores(args.pairwise_scores_file, topics = [args.topic])
  pairwise_scores_df = pairwise_scores_df.rename(columns={args.pairwise_method: "score"})

  predictor_class = kph_method_to_predictor_class[args.tree_method]
  predictor = predictor_class(threshold=args.threshold, pairwise_scores_df=pairwise_scores_df)
  kph = predictor.get_hierarchy()

  gold_dict = load_kph_dict_from_file(args.gold_path)
  evauation_results = Evaluation(gold_dict, {args.topic:kph}, topics=[args.topic]).evaluate_micro()
  logging.info(evauation_results)
  kph_id = f"{args.topic}_{args.threshold}_{args.pairwise_method}_{args.tree_method}"

  os.makedirs(args.output_dir, exist_ok=True)

  tree_path = os.path.join(args.output_dir, f"{kph_id}.jsonl")
  logging.info(f"Writing tree to {tree_path}")

  ordered_kps = get_kps_from_pairwise_df(pairwise_scores_df)
  with jsonlines.open(tree_path, "w") as f:
      kph_json = kph.to_json_line()
      kph_json["kps"] = ordered_kps
      f.write(kph.to_json_line())

  if args.viz:
      viz = kph.get_viz(ordered_kps)
      viz_file = os.path.join(args.output_dir, f"{kph_id}.txt")
      logging.info(f"Writing viz tree to {viz_file}")
      with open(viz_file, "w") as f:
         f.write(viz)
