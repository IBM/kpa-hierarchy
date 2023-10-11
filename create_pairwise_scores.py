import argparse
import os
from conf import thinkp_path
from dataset import Pairwise
from utils import init_logger

if __name__ == "__main__":
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, default=thinkp_path)
    parser.add_argument("--output_path", type=str, help="path of csv to save the pairwise scores")
    args = parser.parse_args()

    data_df = Pairwise(args.gold_path).get_pairs_df()

    ########################################################################################
    # Add your own pairwise scores for each pair in a neW column with a unique method name #
    ########################################################################################

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    data_df.to_csv(args.output_path)

