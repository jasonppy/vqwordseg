import os
import glob
import numpy as np
import json
import tqdm


def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    # while gt_pointer < gt_len and pred_pointer < pred_len:
    #     if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
    #         match_gt += 1
    #         match_pred += 1
    #         gt_pointer += 1
    #         pred_pointer += 1
    #     elif gt[gt_pointer] > pred[pred_pointer]:
    #         pred_pointer += 1
    #     else:
    #         gt_pointer += 1
    for pred_i in pred:
        min_dist = np.abs(gt - pred_i).min()
        match_pred += (min_dist <= tolerance)
    for y_i in gt:
        min_dist = np.abs(pred - y_i).min()
        match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len

def get_word_ali(json_data, index):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    raw_ali = json_data[index].get('text_alignment', None)
    if raw_ali is None:
        return False, False
    
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append([float(toks[0]), float(toks[2])])
    return np.unique(data), True

def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)

vq_wordoutput_dir = "/saltpool0/scratch/pyp/dpdp/data/wordseg/intervals"
with open("/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", "r") as f:
    data_json = json.load(f)['data']
segmentation = {}
match_gt_count = 0
match_pred_count = 0
gt_b_len = 0
pred_b_len = 0
for j, item in enumerate(tqdm.tqdm(data_json)):
    key = item['caption']['wav'].split("/")[-1].split(".")[0]
    prefix = os.path.join(vq_wordoutput_dir, key)
    cur_files = glob.glob(prefix+"*.txt")
    boundaries = []
    for fn in cur_files:
        # print(fn.split("/")[-1].split(".")[0].split("-")[-1])
        s, e = fn.split("/")[-1].split(".")[0].split("-")[-1].split("_")
        s, e = int(s), int(e)
        txtf = open(fn).readlines()
        boundaries += [[(int(item.split(" ")[0])+s)/100., (int(item.split(" ")[1])+s)/100.] for item in txtf]
    boundaries = np.unique(boundaries) # boundary in our format [0.11, 0.24, 0.43, ...] in sec
    gt_boundaries, flag = get_word_ali(data_json, j)
    if not flag:
        continue
    if len(gt_boundaries[1:-1]) == 0:
        continue
    if len(boundaries) <= 2:
        boundaries.append(boundaries[-1])
        boundaries.append(boundaries[-1])
    a, b, c, d = find_boundary_matches(gt_boundaries[1:-1], boundaries[1:-1], tolerance=0.02)
    match_gt_count += a
    match_pred_count += b
    gt_b_len += c
    pred_b_len += d
b_prec = match_pred_count / pred_b_len
b_recall = match_gt_count / gt_b_len
b_f1 = compute_f1(b_prec, b_recall)
b_os = b_recall / b_prec - 1.
b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
print(f"Recall: {b_recall:.4f}")
print(f"Precision: {b_prec:.4f}")
print(f"F1: {b_f1:.4f}")
print(f"OS: {b_os:.4f}")
print(f"R-val: {b_r_val:.4f}")