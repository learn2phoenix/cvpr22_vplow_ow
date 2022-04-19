# Installation
- Create a virtual environment using conda and install pytorch (1.10.0, cuda 10.2), suitable detectron2 and a few required modules
```
conda create -n disc python=3.7
conda activate disc
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
pip install sklearn
```

# Training
To train a detector on pascal voc 2007 train set run the following command
Before begining the training, set correct datapaths in `load_datasets.py`
```
PORT=$((1024+$RANDOM))
DIST_URL=tcp://0.0.0.0:$PORT
TRAIN_DATASET="custom_pascal_voc_trainval_2007"

EXP_NAME="experiments/voc2007_dino_init"
NUM_GPUS=2
python plain_train_net.py --dist-url $DIST_URL --config-file ./configs/faster_rcnn_R_50_C4_3x.yaml --num-gpus $NUM_GPUS  OUTPUT_DIR $EXP_NAME SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 8 DATALOADER.NUM_WORKERS 4 SEED 1235 DATASETS.TRAIN '("'$TRAIN_DATASET'",)' DATASETS.TEST '("custom_pascal_voc_test_2007",)' MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.ROI_HEADS.NAME Res5ROIHeadsExtraNorm MODEL.RESNETS.NORM SyncBN
```
This trains a detector on the PASCAL VOC 2007 set using a Res-50 backbone with DINO weights as supervision. This model achieves an AP50 of `62.50` on the voc 2007 test set (For comparison, a model trained using Imagenet initialization achieves `72.5` AP50 on pascal voc 2007 test set)
Weights for this model can be found in `experiments/voc2007_dino_init/model_final.pth`

# Discovery
- First step is to extract regions from all the images of COCO 2014 train set
- To extract regions, run
```
PORT=$((1024+$RANDOM))
DIST_URL=tcp://0.0.0.0:$PORT
NUM_GPUS=4
python run_detector.py --dist-url $DIST_URL --config-file ./configs/faster_rcnn_R_50_C4_3x.yaml --num-gpus $NUM_GPUS  OUTPUT_DIR experiments/voc2007_dino_init/coco2014_train MODEL.WEIGHTS ./experiments/voc2007_dino_init/model_final.pth DATASETS.TEST '("custom_coco2014_train",)' MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.ROI_HEADS.NAME Res5ROIHeadsExtraNorm MODEL.RESNETS.NORM SyncBN
```
This will take some time as it runs the detector on all 80k images of COCO 2014 trainset and dumps the boxes onto the disk in `experimments/voc2007_dino_init/coco2014_train`.

- This repo performs a simple K-means clustering on all the obtained features for discovery. We use feature extracted from a ViT-S/16 model trained using DINO to obtain the features.
- To extract DINO features
```
python extract_feats.py --data_path <path_to_coco_train> --output_dir ./experiments/voc2007_dino_init/coco2014_train/ --box_file ./experiments/voc20007_dino_init/coco2014-train/preds.pkl
```

- To run clustering and dump discovery results. Here we  use 80 clusters as a baseline
```
python cluster_feats.py --box_file ./experiments/voc2007_dino_init/coco2014_train/preds.pkl --feat_file ./experiments/voc2007_dino_init/coco2014_train/feats_dino.pkl --output_dir ./experiments/voc2007_dino_init/coco2014_train/ --n_cl 80
```
This outputs a csv file, `diiscovery_result.csv` where each row is of the form `img_id,x1,y1,x2,y2,cluster_id`.

- The discovery method can be evaluated using the following command. It computes area under cumulative purity/coverage plots, number of objects discovered and corloc.
```
python discovery_evaluation.py --result_file ./experiments/voc2007_dino_init/coco2014_train/discovery_result.csv --output_dir ./experiments/voc2007_dino_init/coco2014_train/
 ```
This model achieves a AuC of `xx` on the COCO 2014 train set and discovers `yy` objects with a CorLoc of `zz`

- The next step is to train detectors for each of the discovered cluster and test the performance of this detector on COCO minival set
- To extract box predictions on coco minival
```
PORT=$((1024+$RANDOM))
DIST_URL=tcp://0.0.0.0:$PORT
NUM_GPUS=4

python run_detector.py --dist-url $DIST_URL --config-file ./configs/faster_rcnn_R_50_C4_3x.yaml --num-gpus $NUM_GPUS OUTPUT_DIR experiments/voc2007_dino_init/minival MODEL.WEIGHTS ./experiments/voc2007_dino_init/model_final.pth DATASETS.TEST '("custom_coco_minival",)' MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.ROI_HEADS.NAME Res5ROIHeadsExtraNorm MODEL.RESNETS.NORM SyncBN
```

- Next, to get the detections on COCO minival, first extract DINO features of minival detections
```
python extract_feats.py --data_path <path_to_coco_minival> --output_dir ./experiments/voc2007_dino_init/minival/ --box_file ./experiments/voc2007_dino_init/minival/preds.pkl
```
- Finally, mAP50 results on COCO minival can be computed as 
```
python predict_cluster_ids.py --box_file ./experiments/voc2007_dino_init/minival/preds.pkl --feat_file ./experiments/voc2007_dino_init/minival/feats_dino.pkl --model_file ./experiments/voc2007_dino_init/coco2014_train/kmeans_model.pkl --discovery_result_file ./experiments/voc2007_dino_init/coco2014_train/discovery_results.pkl
```
