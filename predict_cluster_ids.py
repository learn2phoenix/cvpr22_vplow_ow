import os
import pickle
import argparse
import json
import pdb
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

def argparser():
    parser = argparse.ArgumentParser(description='Minival eval code')
    parser.add_argument('--box_file', type=str, default='preds.pkl')
    parser.add_argument('--feat_file', type=str, default='feats_dino.pkl')
    parser.add_argument('--model_file', type=str, default='kmeans_model.pkl')

    parser.add_argument('--output_dir', type=str,
                        default='experiments/dummy/minival')
    parser.add_argument('--discovery_result_file',type=str,
                        default='discovery_result.pkl')
    parser.add_argument('--gt_file',type=str,
                        default='data_jsons/instances_coco_minival.json')

    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    # get features
    if not os.path.isfile(args.box_file):
        raise Exception('Box file {} not found'.format(args.box_file))
    if not os.path.isfile(args.feat_file):
        raise Exception('Feature file {} not found'.format(args.feat_file))
    if not os.path.isfile(args.model_file):
        raise Exception('Model file {} not found'.format(args.model_file))
    if not os.path.isfile(args.discovery_result_file):
        raise Exception('Discovery result file {} not found'.format(args.discovery_result_file))
    if not os.path.isfile(args.gt_file):
        raise Exception('GT file {} not found'.format(args.gt_file))

    boxes = pickle.load(open(args.box_file,'rb'))
    feats = pickle.load(open(args.feat_file,'rb'))

    # get kmeans model
    model = pickle.load(open(args.model_file,'rb'))

    # get discovery results for class mapping
    disc_results = pickle.load(open(args.discovery_result_file,'rb'))

    gt = json.load(open(args.gt_file,'rb'))
    class_mapping = {idx: ele['id'] for idx, ele in enumerate(gt['categories'])}
    all_results = []
    all_assign = []
    max_cluster_distances = -np.inf * np.ones((model.n_clusters))
    for k, v in boxes.items():
        cur_feats = np.stack(feats[k])
        cur_assign = model.predict(cur_feats)
        scores = model.transform(cur_feats)
        for ele, assign, score in zip(list(v), cur_assign, scores):
            # coco eval takes inpupt in xywh format
            ele = np.copy(ele)
            ele[2] -= ele[0]
            ele[3] -= ele[1]
            disc_cluster_assign = disc_results['class_name'][assign]
            if disc_cluster_assign == '__background__':
                continue
            coco_class_assign = coco_classes.index(disc_cluster_assign)-1
            coco_category_id = class_mapping[coco_class_assign]
            max_cluster_distances[assign] = max(max_cluster_distances[assign],
                                                score[assign])
            # Write it in detectron output format to use cocoEval seamlessly
            res = {}
            res['image_id'] = k
            res['category_id'] = coco_category_id
            res['bbox'] = ele
            res['score'] = score[assign]
            all_results.append(res)
            all_assign.append(assign)


    for idx, (res, og_assign) in enumerate(zip(all_results, all_assign)):
        # closer to the centroid, the more confident about class
        res['score'] = 1 - res['score']/max_cluster_distances[og_assign]
        all_results[idx] = res

    # evaluate mAP
    cocoGt = COCO(args.gt_file)
    cocoDt = cocoGt.loadRes(all_results)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

