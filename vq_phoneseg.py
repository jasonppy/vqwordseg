#!/usr/bin/env python

"""
Perform phone segmentation on VQ representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from genericpath import exists
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys
import os
import torch
import json
import pickle
from vqwordseg import algorithms


from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature, buildFeature_batch
from scripts.utils.utils_functions import (
    readArgs, writeArgs, loadCPCFeatureMaker, loadClusterModule
    )

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("--feature_dir", type=str, default="/saltpool0/scratch/pyp/dpdp/data/")
    parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json")
    parser.add_argument("--vad", type=str, choices=['python', 'no'], default='python')
    parser.add_argument(
        "--audio_base_path", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/"
    )
    parser.add_argument(
        "--input_format",
        help="format of input VQ representations (default: %(default)s)",
        choices=["npy", "txt"], default="txt"
        )
    parser.add_argument(
        "--algorithm",
        help="VQ segmentation algorithm (default: %(default)s)",
        choices=["dp_penalized", "dp_penalized_n_seg", "dp_penalized_hsmm"],
        default="dp_penalized"
        )
    parser.add_argument(
        "--dur_weight", type=float,
        help="the duration penalty weight; if "
        "not specified, a sensible value is chosen based on the input model",
        default=None
        )
    parser.add_argument(
        "--output_tag", type=str, help="used to name the output directory; "
        "if not specified, the algorithm is used",
        default=None
        )
    parser.add_argument(
        "--downsample_factor", type=int,
        help="factor by which the VQ input is downsampled "
        "(default: %(default)s)",
        default=2
        )
    parser.add_argument(
        "--n_frames_per_segment", type=int,
        help="determines the number of segments for dp_penalized_n_seg "
        "(default: %(default)s)",
        default=7
        )
    parser.add_argument(
        "--n_min_segments", type=int,
        help="sets the minimum number of segments for dp_penalized_n_seg "
        "(default: %(default)s)", default=0
        )
    parser.add_argument(
        "--dur_weight_func",
        choices=["neg_chorowski", "neg_log_poisson", "neg_log_hist",
        "neg_log_gamma"], default="neg_chorowski",
        help="function to use for penalizing duration; "
        "if probabilistic, the negative log of the prior is used"
        )
    parser.add_argument(
        "--model_eos", dest="model_eos", action="store_true",
        help="model end-of-sentence"
        )
    # parser.add_argument(
    #     "--only_save_intervals", dest="only_save_intervals",
    #     action="store_true", help="if set, boundaries and indices are not "
    #     "saved as Numpy archives, only the interval text files are saved"
    #     )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Command-line arguments
    segment_func = getattr(algorithms, args.algorithm)
    dur_weight_func = getattr(algorithms, args.dur_weight_func)
    if args.dur_weight is None:
        if args.model == "vqvae":
            args.dur_weight = 3
        elif args.model == "vqcpc":
            args.dur_weight = 20**2
        elif args.model == "cpc_big":
            args.dur_weight = 3
        else:
            assert False, "cannot set dur_weight automatically for model type"
        if args.algorithm == "dp_penalized_n_seg":
            args.dur_weight = 0
    print(f"Algorithm: {args.algorithm}")
    print(f"Duration weight: {args.dur_weight:.4f}")
    if args.output_tag is None:
        args.output_tag = "phoneseg_{}".format(args.algorithm)

    # # Directories and files
    input_dir = Path(args.feature_dir)
    # z_dir = input_dir/"prequant"
    # print("Reading: {}".format(z_dir))
    # assert z_dir.is_dir(), "missing directory: {}".format(z_dir)
    # if args.input_format == "npy":
    #     z_fn_list = sorted(list(z_dir.glob("*.npy")))
    # elif args.input_format == "txt":
    #     z_fn_list = sorted(list(z_dir.glob("*.txt")))
    # else:
    #     assert False, "invalid input format"

    # # Read embedding matrix
    # embedding_fn = input_dir/"embedding.npy"
    # print("Reading: {}".format(embedding_fn))
    # embedding = np.load(embedding_fn)

    # Segment files one-by-one
    # if not args.only_save_intervals:
    #     boundaries_dict = {}
    #     code_indices_dict = {}
    output_base_dir = input_dir/args.output_tag
    output_base_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_base_dir))
    output_dir = output_base_dir/"intervals"
    output_dir.mkdir(exist_ok=True, parents=True)

    clustering_args = readArgs(
        "/home/pyp/zerospeech2021_baseline/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50_args.json"
        )
    clusterModule = loadClusterModule(
        "/home/pyp/zerospeech2021_baseline/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt"
        )
    clusterModule.cuda()

    # Maybe it's relative path
    if not os.path.isabs(clustering_args.pathCheckpoint):
        clustering_args.pathCheckpoint = os.path.join(
            os.path.dirname(os.path.abspath(
            "/home/pyp/zerospeech2021_baseline/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt")),
            clustering_args.pathCheckpoint
            )
    assert os.path.exists(clustering_args.pathCheckpoint), (
        f"CPC path at {clustering_args.pathCheckpoint} does not exist!!"
        )

    featureMaker = loadCPCFeatureMaker(
        clustering_args.pathCheckpoint, 
        gru_level=vars(clustering_args).get('level_gru', None), 
        get_encoded=clustering_args.encoder_layer, 
        keep_hidden=True)
    featureMaker.eval()
    featureMaker.cuda()

    codebook = clusterModule.Ck.squeeze().cpu().numpy()
    # fn = out_dir/"embedding.npy"
    # print(codebook.shape)
    # print("Writing: {}".format(fn))
    # np.save(fn, codebook)

    def cpc_feature_function(x): 
        return buildFeature(
            featureMaker, x, seqNorm=False, strict=True, maxSizeSeq=10240
            )

    with open(args.data_json, "r") as f:
        data_json = json.load(f)['data']
    # load vad file
    vad_json_fn = args.data_json.split(".")[0] + "_with_alignments"+ "_vad_" + args.vad  + ".json"
    with open(vad_json_fn, "r") as f:
        vad_json = json.load(f)
    vad_dict = {cur_key.split("/")[-1].split(".")[0]:vad_json[cur_key] for cur_key in vad_json}
    
    all_seg = {"data": {}, "inter": {}}
    # for path in tqdm(sorted(list(dataset_dir.rglob("*.flac")))):
    for j, item in enumerate(tqdm(data_json)):
        # print(path, output_format)
        path = Path(os.path.join(args.audio_base_path, item['caption']['wav']))
        all_seg['inter'][path.stem] = []
        # out_path = (out_dir / path.stem).with_suffix("")
        # out_path.parent.mkdir(exist_ok=True, parents=True)

        features = cpc_feature_function(path).cuda()
        codes = torch.argmin(clusterModule(features), dim=-1)
        codes = codes[0].detach().cpu().numpy()

        z = features[0].detach().cpu().numpy()

        cur_vad = vad_dict[path.stem]
        # last_interval = cur_vad[-1]
        # Read pre-quantisation representations
        # if args.input_format == "npy":
        #     z = np.load(path)
        # elif args.input_format == "txt":
        #     z = np.loadtxt(path)
        # key = path.stem # m15yw127w9r2pl-3JNQLM5FT4MUW3MAGP9EAXRWP7R2LH_364521_342214
        # Segment
        if z.ndim == 1:
            # print(input_fn)
            # assert False
            continue
        # if len(cur_vad) == 1:
        #     continue
        for cur_interval in cur_vad:
            
            cur_s, cur_e = int(cur_interval[0]/args.downsample_factor), int(cur_interval[1]/args.downsample_factor)
            cur_z = z[cur_s: cur_e+1]
            if args.algorithm == "dp_penalized_n_seg":
                _, code_indices = segment_func(
                    codebook, cur_z, dur_weight=args.dur_weight,
                    n_frames_per_segment=args.n_frames_per_segment,
                    n_min_segments=args.n_min_segments,
                    dur_weight_func=dur_weight_func
                    )
            else:
                _, code_indices = segment_func(
                    codebook, cur_z, dur_weight=args.dur_weight,
                    dur_weight_func=dur_weight_func, model_eos=args.model_eos
                    )

            # Convert boundaries to same frequency as reference
            if args.downsample_factor > 1:
                # boundaries_upsampled = np.zeros(
                #     len(boundaries)*args.downsample_factor, dtype=bool
                #     )
                # for i, bound in enumerate(boundaries):
                #     boundaries_upsampled[i*args.downsample_factor + 1] = bound
                # boundaries = boundaries_upsampled

                code_indices_upsampled = []
                for start, end, index in code_indices:
                    code_indices_upsampled.append((
                        start*args.downsample_factor, 
                        end*args.downsample_factor,
                        index
                        ))
                code_indices = code_indices_upsampled

            # Merge repeated codes (only possible for intervals > 15 frames)
            i_token = 0
            while i_token < len(code_indices) - 1:
                cur_start, cur_end, cur_label = code_indices[i_token]
                next_start, next_end, next_label = code_indices[i_token + 1]
                if cur_label == next_label:
                    code_indices.pop(i_token)
                    code_indices.pop(i_token)
                    code_indices.insert(
                        i_token,
                        (cur_start, next_end, cur_label)
                        )
                    # print(input_fn.stem, cur_start, next_end, cur_label)
                else:
                    i_token += 1

            # Write intervals
            all_seg["data"][path.stem+f"-{str(cur_interval[0])}_{str(cur_interval[1])}"] = code_indices
            all_seg["inter"][path.stem].append(f"-{str(cur_interval[0])}_{str(cur_interval[1])}")
            # with open((output_dir/(utt_key + f"-{str(cur_interval[0])}_{str(cur_interval[1])}")).with_suffix(".txt"), "w") as f:
            #     for start, end, index in code_indices:
            #         f.write("{:d} {:d} {:d}\n".format(start, end, index))

            # if not args.only_save_intervals:
            #     boundaries_dict[utt_key] = boundaries
            #     code_indices_dict[utt_key] = code_indices
    #     if j >= 10:
    #         break
    # print(all_seg)
    # if not args.only_save_intervals:

    #     # Write code indices
    #     output_fn = output_base_dir/"indices.npz"
    #     print("Writing: {}".format(output_fn))
    #     np.savez_compressed(output_fn, **code_indices_dict)

    #     # Write boundaries
    #     output_fn = output_base_dir/"boundaries.npz"
    #     print("Writing: {}".format(output_fn))
    #     np.savez_compressed(output_fn, **boundaries_dict)
    print(f"save phone segmentation at {output_dir/'phoneseg_dict.pkl'}")
    with open(output_dir/"phoneseg_dict.pkl", "wb") as f:
        pickle.dump(all_seg, f)

if __name__ == "__main__":
    main()
