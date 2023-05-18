import os
import time
import datetime
import json
import pdb
import sys
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import argparse
from tqdm import tqdm
import pickle
# import swifter
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial

def argparser():
    parser = argparse.ArgumentParser(description='Eval code')
    parser.add_argument('--result_file', type=str,
                        default='discovery_results.csv')
    parser.add_argument('--output_dir', type=str,
                        default='./')
    parser.add_argument('--num_workers', type=int,
                        default=4)

    return parser.parse_args()


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

pascal_indices = [2,5,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
pascal_classes = [coco_classes[i] for i in pascal_indices]

def get_auc(purity, coverage):
    coverage_append = [0] + coverage
    coverage_diff = np.diff(coverage_append)
    return np.sum(purity*coverage_diff)

def get_knwn_sorted_purity_coverage(dump, marker):

    pur = [dump['res']['purity'][i] for i in range(marker)
           if dump['res']['class_name'][i] != '__background__'
           and dump['res']['class_name'][i] in pascal_classes]

    cov = [dump['res']['coverage'][i][1][np.in1d(np.arange(81), pascal_indices)]
           for i in range(marker)
           if dump['res']['class_name'][i] != '__background__'
           and dump['res']['class_name'][i] in pascal_classes]

    pur_sort = np.sort(pur)[::-1]
    pur_sort_ord = np.argsort(pur)[::-1]
    cov_sort = [cov[i] for i in pur_sort_ord]

    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort)+1)]

    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
                for i in range(1, len(cov_sort)+1)]
    return pur_cum, cov_cum


def get_purity_coverage(dump, marker):

    pur = [dump['res']['purity'][i] for i in range(marker)
           if dump['res']['class_name'][i] != '__background__']

    cov = [dump['res']['coverage'][i][1] for i in range(marker) if dump['res']['class_name'][i] !=
           '__background__']
    objs = [dump['res']['class_name'][i] for i in range(marker)
            if dump['res']['class_name'][i] != '__background__']
    pur_sort = np.sort(pur)[::-1]
    pur_sort_ord = np.argsort(pur)[::-1]
    cov_sort = [cov[i] for i in pur_sort_ord]
    obj_sort = [objs[i] for i in pur_sort_ord]

    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort)+1)]
    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
               for i in range(1, len(cov_sort)+1)]
    obj_cum = [len(list(set(obj_sort[:i]))) for i in range(1, len(obj_sort)+1)]
    return pur_cum, cov_cum, obj_cum,


def get_novel_sorted_purity_coverage(dump, marker, cluster_limit):

    pur = [dump['res']['purity'][i] for i in range(marker)
           if dump['res']['class_name'][i] != '__background__'
           and dump['res']['class_name'][i] not in pascal_classes]

    cov = [dump['res']['coverage'][i][1][~np.in1d(np.arange(81), pascal_indices)]
           for i in range(marker) if dump['res']['class_name'][i] != '__background__'
           and dump['res']['class_name'][i] not in pascal_classes]
    objs = [dump['res']['class_name'][i] for i in range(marker)
            if dump['res']['class_name'][i] != '__background__'
            and dump['res']['class_name'][i] not in pascal_classes]
    pur_sort = np.sort(pur)[::-1][:cluster_limit]
    pur_sort_ord = np.argsort(pur)[::-1][:cluster_limit]
    cov_sort = [cov[i] for i in pur_sort_ord]
    obj_sort = [objs[i] for i in pur_sort_ord]

    pur_cum = [np.mean(pur_sort[:i]) for i in range(1, len(pur_sort)+1)]
    cov_cum = [np.mean(np.sum(np.stack(cov_sort[:i], axis=0), axis=0), axis=0)
               for i in range(1, len(cov_sort)+1)]
    obj_cum = [len(list(set(obj_sort[:i]))) for i in range(1, len(obj_sort)+1)]
    return pur_cum, cov_cum, obj_cum,


def find_iou(box1, box2):
    """ Find iou """
    # box1 is of shape Nx4 [x1,y1,x2,y2]
    # box2 is of shape Mx4 [x1,y1,x2,y2]
    # returns a NxM IoU matrix
    box1 = np.expand_dims(box1, axis=1) # shape Nx1x4
    box2 = np.expand_dims(box2, axis=0) # shape 1xMx4
    I_x = np.minimum(box1[:,:,2], box2[:,:,2]) - np.maximum(box1[:,:,0], box2[:,:,0])
    I_y = np.minimum(box1[:,:,3], box2[:,:,3]) - np.maximum(box1[:,:,1],
                                                           box2[:,:,1])
    I_x[I_x<0] = 0
    I_y[I_y<0] = 0
    I_A = I_x * I_y

    A_1 = (box1[:,:,2] - box1[:,:,0])*(box1[:,:,3] - box1[:,:,1])
    A_2 = (box2[:,:,2] - box2[:,:,0])*(box2[:,:,3] - box2[:,:,1])
    U_A = A_1 + A_2 - I_A
    return I_A/U_A


def eval_variables(cache_meta):
    # initialize flags
    cluster_flag_dict = {}
    cluster_pos_label_dict = {}
    coverage_cluster_dict = {}
    dump_data = {}
    coverage_dict = {}
    # Get variables for eval
    for key, value in cache_meta.items():
        dump_data[key] = -1*np.ones(len(value))
        cluster_flag_dict[key] = -1*np.ones(len(value))
        cluster_pos_label_dict[key] = np.zeros(len(value))
        coverage_cluster_dict[key] = {}
        coverage_dict[key] = []
        for sample_num, sample in enumerate(value):
            coverage_cluster_dict[key][sample_num] = {}
    
    return cluster_flag_dict, cluster_pos_label_dict, coverage_cluster_dict, dump_data, coverage_dict

def process_cluster_cache(cluster, cache_meta, cluster_flag_dict,
                        cluster_pos_label_dict, coverage_cluster_dict,
                        coverage_dict, box_counts, num_img_threshold=5):
    results = {}
    cur_pos_list = cluster_pos_label_dict[cluster]
    cur_flag = cluster_flag_dict[cluster]
    # Values in flag > 0 have IoUs > thresh with some groound truth
    num_positives = float(np.sum(cur_flag >= 0))
    coverage_meta = coverage_cluster_dict[cluster]
    new_coverage_meta = coverage_dict[cluster]
    coverage_list_stuff = new_coverage_meta.copy()
    results['id_check'] = 0
    results['num_samples_in_cluster'] = len(coverage_list_stuff)
    
    if len(cache_meta[cluster]) != len(coverage_meta):
        pdb.set_trace()

    if num_positives > 0:

        if np.all(cur_pos_list <= 0):
            results['purity'] = 0
            results['coverage'] = [np.zeros(81), 0]
            results['class_name'] = coco_classes[0]
            results['pos_id'] = cur_flag
            results['pos_label'] = cur_pos_list
            results['cov_assign'] = new_coverage_meta
            results['cag_coverage'] = 0

        else:
            cluster_stuff = np.array(coverage_list_stuff)
            if len(cluster_stuff.shape)>1:
                results['id_check'] = np.unique(cluster_stuff[:,0]).shape[0]
                results['num_samples_in_cluster'] = len(coverage_list_stuff)
            else:
                results['id_check'] = -1
                results['num_samples_in_cluster'] = -1
            
            # discard clusters which do not span across 'num_img_threshold' images
            if results['id_check'] < num_img_threshold:
                results['purity'] = 0
                results['coverage'] = [np.zeros(81), 0]
                results['class_name'] = coco_classes[0]
                results['pos_id'] = cur_flag
                results['pos_label'] = cur_pos_list
                results['cov_assign'] = new_coverage_meta
                results['cag_coverage'] = 0
            else:
                class_dist_counter = Counter(cur_pos_list[cur_flag>=0])

                class_dist = class_dist_counter.most_common(1)[0]
                # Coverage is like recall, so don't consider duplicates
                # consider only positives hence cur_flag > 0
                # Coverage is number of gt boxes covered by crrent cluster
                # class_dist_counter_coverage = Counter(cur_pos_list[cur_flag > 0])
                class_agnostic_coverage = len(new_coverage_meta)/np.sum(box_counts[1:])

                coverage_arr = np.zeros(81)
                for ele in new_coverage_meta:
                    coverage_arr[int(ele[2])] += 1
                if np.sum(coverage_arr) != len(new_coverage_meta):
                    pdb.set_trace()
                coverage_arr = coverage_arr.astype(float)
                coverage_arr = [0] + list((coverage_arr[1:])/(box_counts[1:]+1e-08))
                coverage_arr = np.asarray(coverage_arr)

                coverage_arr[(box_counts == 0).astype(bool)] = 0
                if not np.all(coverage_arr <= 1.):
                    assert np.all(coverage_arr <= 1.)
                try:
                    pos_smpls = np.where(cur_pos_list == class_dist[0])[0]

                except:
                    pdb.set_trace()
                if np.isnan(class_dist[1]/num_positives):
                    pdb.set_trace()
                if class_dist[1]/float(cur_pos_list.shape[0]) > 1:
                    pdb.set_trace()
                results['purity'] = class_dist[1]/float(cur_pos_list.shape[0])
                results['class_name'] = coco_classes[int(class_dist[0])]
                results['pos_id'] = cur_flag
                results['pos_label'] = cur_pos_list
                results['cov_assign'] = new_coverage_meta
                results['cag_coverage'] = len(new_coverage_meta)/box_counts.sum()
                cvrg = np.zeros(81)
                for ele in new_coverage_meta:
                    cvrg[int(ele[2])] += 1
                final_coverage = np.asarray([0] + (cvrg[1:]/(box_counts[1:] + 1e-08)).tolist())
                if np.any(final_coverage > 1):
                    pdb.set_trace()
                results['coverage'] = [0, final_coverage, 0, 0]
        

    else:
        results['purity'] = 0
        results['coverage'] = [0, np.zeros(81), 0, 0]
        results['class_name'] = coco_classes[0]
        results['pos_id'] = cur_flag
        results['pos_label'] = cur_pos_list
        results['cov_assign'] = new_coverage_meta
        results['cag_coverage'] = 0
    cvrg = np.zeros(81)
    # import pdb;pdb.set_trace()
    # for ele in coverage_list:
    #     cvrg[int(ele[2])] += 1
    # final_coverage = np.asarray([0] + (cvrg[1:]/(box_counts[1:] + 1e-08)).tolist())

    # results['final_coverage'] = final_coverage
    return results, coverage_list_stuff

def log(logger, str):
    if logger:
        logger.info(str)
        
def eval_discovery_metrics(roi_dump, gt_dump, cache_meta, all_labels,
                           iou_thresh=0.5, marker=80, verbose=False,
                           cluster_limits=[100,200,500,1000], num_workers=4, logger=None):
    # for coco... Hard coded for now
    box_counts = np.zeros(81)
    cluster_flag_dict, cluster_pos_label_dict, coverage_cluster_dict, dump_data, coverage_dict = eval_variables(cache_meta)

    start = 0.0
    end = 0.0
    avg = 0.0
    elapsed = 0.0
    im_names = list(gt_dump.keys())
    corloc_count = 0
    covered_count = 0
    total_boxes = 0.
    for idx, val in enumerate(im_names):
        start = time.time()
        gt_boxes = gt_dump[val]
        rem = []
        #### THIS CAN BE OPTIMIZED
        total_boxes += len(gt_boxes)
        for id_box, box in enumerate(gt_boxes):
            if box[4] <= 80:
                box_counts[int(box[4])] += 1
            else:
                rem.append(id_box)
                pdb.set_trace()
        proposal_ious = []
        proposal_meta = []
        if val not in all_labels:
            continue
        cluster_assignments = all_labels[val]
        grp_roi = []
        for lab_id, lab in enumerate(cluster_assignments):
            if not np.isnan(lab):
                if lab >= 0:
                    try:
                        # import pdb;pdb.set_trace()
                        sample_id = cache_meta[lab].index((val, lab_id))
                    except:
                        pdb.set_trace()
                    cluster_num = lab
                    feat_idx = lab_id
                    grp_roi.append(roi_dump[val][feat_idx])
                    proposal_meta.append([sample_id, cluster_num, idx,
                                          val, feat_idx])
        if not grp_roi:
            # if no predictions in current image
            continue
        grp_roi = np.stack(grp_roi, axis=0)
        proposal_ious = find_iou(grp_roi, gt_boxes[:, :4])
        if (proposal_ious >= 0.5).any():
            corloc_count += 1

        iou = proposal_ious
        visited_proposal = np.zeros(iou.shape[0], dtype=bool)
        gts_covered = np.zeros((iou.shape[1]))
        for gt_box_id in range(iou.shape[1]):
            cur_ious = iou[:,gt_box_id]
            # Sorted for no apparent reason
            sorted_ious = np.sort(cur_ious)[::-1]
            sorted_ious_ids = np.argsort(cur_ious)[::-1]
            found = False
            for ii, (iou_, corr_id) in enumerate(zip(sorted_ious,
                                                     sorted_ious_ids)):
                cluster_num_sample = proposal_meta[corr_id][1]

                cur_sample_id = proposal_meta[corr_id][0]

                if iou_ < iou_thresh:
                    # If iou is less than threshold, then don't bother processing all other sorted proposals
                    if dump_data[cluster_num_sample][cur_sample_id] == -1:
                        dump_data[cluster_num_sample][cur_sample_id] = iou_
                    break
                if not found and not visited_proposal[corr_id]:
                    cluster_pos_label_dict[cluster_num_sample][cur_sample_id] = gt_boxes[gt_box_id, 4]
                    cluster_flag_dict[cluster_num_sample][cur_sample_id] = 1
                    dump_data[cluster_num_sample][cur_sample_id] = iou_
                    coverage_cluster_dict[cluster_num_sample][cur_sample_id] = (proposal_meta[corr_id][2], gt_box_id)
                    if (idx, gt_box_id, gt_boxes[gt_box_id, 4]) not in coverage_dict[cluster_num_sample]:
                        coverage_dict[cluster_num_sample].append((idx, gt_box_id, gt_boxes[gt_box_id, 4]))
                    found = True
                    visited_proposal[corr_id] = True
                    gts_covered[ii] = 1
                else:
                    cluster_pos_label_dict[cluster_num_sample][cur_sample_id] = gt_boxes[gt_box_id, 4]
                    cluster_flag_dict[cluster_num_sample][cur_sample_id] = 0
                    dump_data[cluster_num_sample][cur_sample_id] = iou_
                    coverage_cluster_dict[cluster_num_sample][cur_sample_id] = (proposal_meta[corr_id][2], gt_box_id)
        covered_count += gts_covered.sum()
        end = time.time() - start
        elapsed += end
        avg = ((idx*avg) + end)/(idx+1)
        if verbose:
            if idx %1000 == 0:
                sys.stdout.write("\033[F\033[K")
                print('Image {}/{}, Elapsed Time : {} hh:mm:ss, Remaining Time : {} hh:mm:ss'.format(idx+1, len(im_names), datetime.timedelta(seconds=(elapsed)),
                                datetime.timedelta(seconds=((len(im_names)-idx+1)*avg))))

                
                

    results = defaultdict(list)
    results['corloc'] = corloc_count/len(gt_dump)
    results['recall'] = covered_count/box_counts.sum()
    results['gt_box_cnts'] = box_counts
    results['coverage_dict'] = coverage_dict
    coverage_list = []
    s1 = time.time()
    func_to_run = partial(process_cluster_cache, coverage_cluster_dict=coverage_cluster_dict,
                        cluster_pos_label_dict=cluster_pos_label_dict, 
                        cluster_flag_dict=cluster_flag_dict, 
                        coverage_dict=coverage_dict,
                        box_counts=box_counts,
                        cache_meta=cache_meta
                        )
    print(f'Using {num_workers} workers')
    log(logger, f'Using {num_workers} workers')

   
    with mp.Pool(processes=num_workers) as pool:
        mp_results = []
        idx = 0
        for some_result in pool.map(func_to_run, list(cache_meta.keys())):
            result_retuerned, coverage_stuff = some_result
            for key, val in result_retuerned.items():
                results[key].append(val)
            coverage_list += coverage_stuff
    # New metric calualtion. 
    results["new_metric"] = np.sum(np.array(results["purity"]) * np.array(results["num_samples_in_cluster"])) / total_boxes
    s2 = time.time()
    print("For loop ", s2-s1)
    
    print('**'*10)
    print('**'*10)
    knwn_cum_pur, knwn_cum_cov = get_knwn_sorted_purity_coverage({'res': results}, marker)
    knwn_auc = get_auc(knwn_cum_pur, knwn_cum_cov)
    for cluster_limit in cluster_limits + ['full']:
        if cluster_limit == 'full':
            cluster_limit = marker
            extra_str = ''
        else:
            extra_str = f'_{cluster_limit}'

        unknwn_cum_pur, unknwn_cum_cov, unknwn_num_obj = get_novel_sorted_purity_coverage({'res': results}, marker, cluster_limit)
        unknwn_auc = get_auc(unknwn_cum_pur, unknwn_cum_cov)
        print('**'*10)
        print('**'*10)
        num_unknwn_obj = 0 if len(unknwn_num_obj) == 0 else unknwn_num_obj[-1]
        print('Unknown AUC(@{}): {}, Discovered Objects: {}, Number of Clusters: {}'.format(iou_thresh, unknwn_auc,
            num_unknwn_obj, cluster_limit))
        print('CorLoc: {}'.format(results['corloc']))
        results[f'unknown_auc{extra_str}'] = unknwn_auc
        results[f'unknown_cum_pur{extra_str}'] = unknwn_cum_pur
        results[f'unknown_cum_cov{extra_str}'] = unknwn_cum_cov
        results[f'unknown_num_obj{extra_str}'] = num_unknwn_obj

    results['number_of_clusters'] = marker

    results['known_auc'] = knwn_auc
    results['known_cum_pur'] = knwn_cum_pur
    results['known_cum_cov'] = knwn_cum_cov

    results['iou'] = [iou_thresh]
    print("Remaining ", time.time()-s2)
    return results, dump_data

# @profile
def create_gt():
    # Load ground truth file
    gt_dump = {}
    with open('/vulcanscratch/anubhav/AbhinavLab/code/cvpr22_vplow_ow/data_jsons/instances_train2014.json', 'rb') as f:
        gt = json.load(f)

    gt_class_mapping = {ele['id']: idx + 1 for idx, ele in
                        enumerate(gt['categories'])}
    for anno in tqdm(gt['annotations']):
        if anno['image_id'] not in gt_dump:
            gt_dump[anno['image_id']] = []
        cur_box = anno['bbox']
        # box is in xywh format
        cur_box[2] += cur_box[0]
        cur_box[3] += cur_box[1]
        gt_dump[anno['image_id']].append(np.asarray(cur_box +
                                                    [gt_class_mapping[anno['category_id']]]))

    for k, v in gt_dump.items():
        gt_dump[k] = np.stack(v)
        assert (gt_dump[k][:, 4] > 0).all()
    
    return gt_dump

# @profile
def read_result(result_file):

    def merge(idxs, img_ids):
        return [(img_id, idx) for img_id, idx in zip(img_ids, idxs)]

    # read the  csv file
    results = pd.read_csv(result_file, header=0)
    cache_meta = defaultdict(list)
    all_labels = {}
    roi_dump = {}
    results_by_image = results.groupby('image_id')
    results['idx'] = results.groupby('image_id').cumcount()
    cache_meta = results.groupby('cluster_id').apply(lambda x : merge(list(x['idx']), list(x['image_id']))).to_dict()
    roi_dump = results_by_image.apply(lambda x : np.stack([x['x1'], x['y1'], x['x2'], x['y2']], axis=-1)).to_dict()

    for img_id, res in tqdm(results_by_image):
        if img_id not in all_labels:
            all_labels[img_id] = -np.inf * np.ones((res.shape[0]))
        all_labels[img_id][res['idx']] = res['cluster_id']
        
    return cache_meta, all_labels, roi_dump

def main(result_file, output_dir):

    cache_meta, all_labels, roi_dump = read_result(result_file)
    
    gt_dump = create_gt()

    # import pdb;pdb.set_trace()
    
    discovery_result = eval_discovery_metrics(roi_dump, gt_dump, cache_meta,
                                              all_labels,
                                              marker=int(len(cache_meta)),
                                              verbose=True)
    with open(os.path.join(output_dir, 'discovery_results.pkl'), 'wb') as f:
        pickle.dump(discovery_result[0], f)


if __name__ == '__main__':
    args = argparser()
    start = time.time()
    main(args.result_file, args.output_dir)
    print("Program Ends : ", time.time() - start)


