#!/usr/bin/env python

"""
Perform word segmentation on VQ representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

from vqwordseg import algorithms
import eval_segmentation


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("--feature_dir", type=str, default="/saltpool0/scratch/pyp/dpdp/data/")
    parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
    parser.add_argument("--vad", type=str, choices=['python', 'no'], default='python')
    parser.add_argument(
        "--algorithm",
        help="word segmentation algorithm (default: %(default)s)",
        choices=["ag", "tp", "rasanen15", "dpdp_aernn"], default="ag"
        )
    parser.add_argument(
        "--output_tag", type=str, help="used to name the output directory; "
        "if not specified, the algorithm is used",
        default=None
        )
    parser.add_argument(
        "--dur_weight", type=float,
        help="the duration penalty weight",
        default=None
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    # Command-line arguments
    segment_func = getattr(algorithms, args.algorithm)
    args.output_tag = "wordseg"
    if args.dur_weight is not None:
        print(f"Duration weight: {args.dur_weight:.4f}")

    # Phone intervals
    input_dir = (
        Path(args.feature_dir)/"phoneseg_dp_penalized/intervals"
        )
    phoneseg_interval_dict = {}
    print("Reading: {}".format(input_dir))
    assert input_dir.is_dir(), "missing directory: {}".format(input_dir)
    phoneseg_interval_dict = eval_segmentation.get_intervals_from_dir(
        input_dir
        )
    utterances = phoneseg_interval_dict.keys()

    # # Temp
    # print(list(utterances)[228], list(utterances)[5569])
    # assert False

    # Segmentation
    print(datetime.now())
    print("Segmenting:")
    prepared_text = []
    for utt_key in utterances:
        prepared_text.append(
            " ".join([i[2] + "_" for i in phoneseg_interval_dict[utt_key]])
            )
    if args.dur_weight is not None:
        word_segmentation = segment_func(
            prepared_text, dur_weight=args.dur_weight
            )
    else:
        word_segmentation = segment_func(
            prepared_text
            )
    print(datetime.now())
    # print(prepared_text[:10])
    # print(word_segmentation[:10])
    # assert False
    wordseg_interval_dict = {}
    for i_utt, utt_key in tqdm(enumerate(utterances)):
        words_segmented = word_segmentation[i_utt].split(" ")
        word_start = 0
        word_label = ""
        i_word = 0
        wordseg_interval_dict[utt_key] = []
        for (phone_start,
                phone_end, phone_label) in phoneseg_interval_dict[utt_key]:
            word_label += phone_label + "_"
            if i_word >= len(words_segmented):
                wordseg_interval_dict[utt_key].append((
                    word_start, phoneseg_interval_dict[utt_key][-1][1],
                    "999_" #word_label
                    ))
                break
            if words_segmented[i_word] == word_label:
                wordseg_interval_dict[utt_key].append((
                    word_start, phone_end, word_label
                    ))
                word_label = ""
                word_start = phone_end
                i_word += 1

    # Write intervals
    output_dir = (
        Path(args.feature_dir)/args.output_tag/"intervals"
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(wordseg_interval_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, label in wordseg_interval_dict[utt_key]:
                f.write("{:d} {:d} {}\n".format(start, end, label))


if __name__ == "__main__":
    main()
