import argparse
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans


def argparser():
    parser = argparse.ArgumentParser(description='Clustering code')
    parser.add_argument('--box_file', type=str, default='preds.pkl')
    parser.add_argument('--feat_file', type=str, default='feats_dino.pkl')
    parser.add_argument('--output_dir', type=str, default='experiments/dummy')
    parser.add_argument('--n_cl', type=int, default=20)
    return parser.parse_args()


def dump_output(boxes, labels, ids, output_dir=None):
    # First convert to correct output format and then dump to disk
    # output format id,x1,y1,x2,y2,label
    assert boxes.shape[1] == 4
    assert labels.shape[1] == 1
    assert ids.shape[1] == 1

    output = np.concatenate([ids, boxes, labels], axis=1)
    df = pd.DataFrame(output,
                      columns=['image_id', 'x1', 'y1', 'x2', 'y2', 'cluster_id'])
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, 'discovery_result.csv'), index=None)
    return df


def main(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # read 
    box_file = args.box_file
    feat_file = args.feat_file
    with open(box_file, 'rb') as f:
        boxes = pickle.load(f)
    with open(feat_file, 'rb') as f:
        feats = pickle.load(f)

    all_feats = []
    all_boxes = []
    all_ids = []

    for key, val in boxes.items():
        if not len(val):
            continue
        if not isinstance(val, list):
            val = [val]

        all_feats += [np.stack(feats[key])]
        # boxes are in xyxy format
        all_boxes += [np.stack(val)]
        all_ids.extend([np.asarray([key] * len(val))])
    assert (np.asarray([l.shape[0] for l in all_feats]) == np.asarray([l.shape[0] for l in all_boxes])).all()
    assert (np.asarray([l.shape[0] for l in all_ids]) == np.asarray([l.shape[0]
                                                                     for l in
                                                                     all_boxes])).all()

    all_feats = np.concatenate(all_feats)
    all_boxes = np.concatenate(all_boxes)
    all_ids = np.concatenate(all_ids)
    if all_ids.ndim == 1:
        all_ids = all_ids[:, None]

    if all_feats.shape[0] < 200000:
        kmeans = KMeans(n_clusters=args.n_cl, random_state=0, verbose=True).fit(all_feats)
        kmeans_labels = kmeans.labels_
    else:
        # if there are a lot of features, use subset of them and do label
        # propagation
        print('Using only subset of feats')
        req_feats_ids = np.random.choice(np.arange(all_feats.shape[0]), size=10000,
                                         replace=False)
        req_feats = all_feats[req_feats_ids, :]
        kmeans_labels = np.zeros((all_feats.shape[0]))
        kmeans = KMeans(n_clusters=args.n_cl, random_state=0,
                        verbose=True).fit(req_feats)
        rem_feats_ids = np.where(~np.in1d(np.arange(all_feats.shape[0]),
                                          req_feats_ids))[0]
        rem_feats = all_feats[rem_feats_ids, :]
        kmeans_labels[req_feats_ids] = kmeans.labels_
        kmeans_labels[rem_feats_ids] = kmeans.predict(rem_feats)

    kmeans_labels = kmeans_labels.astype(int)
    if kmeans_labels.ndim == 1:
        kmeans_labels = kmeans_labels[:, None]
    _ = dump_output(all_boxes, kmeans_labels, all_ids,
                    output_dir=args.output_dir)
    with open(os.path.join(args.output_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)


if __name__ == '__main__':
    args = argparser()
    main(args)
