import os
import pickle
import argparse
import json
import pdb

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np

coco_classes = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')


def eval_detection_metrics(det_csv_file, gt_file, mapping_needed = False, disc_results = None):
    if mapping_needed:
        assert disc_results != None, "Discovery results is needed"
        gt = json.load(open(gt_file,'rb'))
        class_mapping = {idx: ele['id'] for idx, ele in enumerate(gt['categories'])}
    pred_df = pd.read_csv(det_csv_file)
    all_results = []
    #import pdb
    #pdb.set_trace()
    for idx, row in pred_df.iterrows():
        res = {}

        res['image_id'] = row['image_id']
        if int(row['cluster_id']) < 0 or int(row['cluster_id']) >= 80 :
            continue
        if mapping_needed:
            #print(f'cluster_id : {int(row["cluster_id"])}')
            disc_cluster_assign = disc_results['class_name'][int(row['cluster_id'])]
            coco_class_assign = coco_classes.index(disc_cluster_assign)-1
            coco_category_id = class_mapping[coco_class_assign]
        else:
            coco_category_id = int(row['cluster_id'])
        res['category_id'] = coco_category_id
        row['w'] = row['x2'] - row['x1']
        row['h'] = row['y2'] - row['y1']
        res['bbox'] = [row['x1'], row['y1'], row['w'], row['h']]
        res['score'] = row['conf_score']

        all_results.append(res)
        
    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(all_results)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0], cocoEval.stats[1]


def argparser():
    parser = argparse.ArgumentParser(description='Det Eval code')
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--output_dir', type=str,
                        default='./')
    parser.add_argument('--discovery_results', type=str)
    parser.add_argument('--gt_file', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    import pickle
    discovery_results = pickle.load(open(args.discovery_results, 'rb'))
    detection_result = eval_detection_metrics(det_csv_file=args.result_file,
                                                 gt_file =args.gt_file,
                                                 mapping_needed = True,
                                                 disc_results = discovery_results)
    print(detection_result)

