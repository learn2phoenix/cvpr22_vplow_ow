# Installation
- Create a virtual environment using conda and install pytorch (1.10.0, cuda 10.2), suitable detectron2 and a few required modules
```
conda create -n disc python=3.7
conda activate disc
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
pip install sklearn
pip install opencv-python
pip install scikit-image
pip install pandas
```

# Downloads
The pretrained backbones and baselines can be downloaded from the following links. The first 2 should be placed in the `weights` folder for default commands to work:
* [ResNet-50 backbone w/o imagenet](https://drive.google.com/file/d/1izgsq_XvziyPf6tZy19PchXJPh4UVtYK/view?usp=sharing)
* [ResNet-50 backbone w imagenet](https://drive.google.com/file/d/1v_ZsPDcMhVp1hTaKs73Kk3AI23fXATkW/view?usp=sharing)
* [PASCAL VOC 2007 baseline w/o image](https://drive.google.com/file/d/1R-VbFrDB2RHQ4m5GUw2sA41w4lUbZ1IN/view?usp=sharing)
* [PASCAL VOC 2007 baseline w image](https://drive.google.com/file/d/1Xs56QTWYjFawwl2tE5dkk9r41i6zpF7g/view?usp=sharing)


# Training
To train a detector on pascal voc 2007 train set with DINO weights as initialization run the following command
Before begining the training, set correct datapaths in `load_datasets.py`
```
PORT=$((1024+$RANDOM))
DIST_URL=tcp://0.0.0.0:$PORT
TRAIN_DATASET="custom_pascal_voc_trainval_2007"

EXP_NAME="experiments/voc2007_dino_init"
NUM_GPUS=2
python plain_train_net.py --dist-url $DIST_URL --config-file ./configs/faster_rcnn_R_50_C4_3x.yaml --num-gpus $NUM_GPUS  OUTPUT_DIR $EXP_NAME SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 8 DATALOADER.NUM_WORKERS 4 SEED 1235 DATASETS.TRAIN '("'$TRAIN_DATASET'",)' DATASETS.TEST '("custom_pascal_voc_test_2007",)' MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.ROI_HEADS.NAME Res5ROIHeadsExtraNorm MODEL.RESNETS.NORM SyncBN MODEL.WEIGHTS weights/dino_resnet50_pretrain_changed.pth
```
This trains a detector on the PASCAL VOC 2007 set using a Res-50 backbone with DINO weights as initialization. This model achieves an AP50 of `62.50` on the voc 2007 test set (For comparison, a model trained using Imagenet initialization achieves `72.5` AP50 on pascal voc 2007 test set)
Weights for this model can be found in `experiments/voc2007_dino_init/model_final.pth`

To train a detector on pascal voc 2007 train set with imagenet initialization run the following command
Before begining the training, set correct datapaths in `load_datasets.py`
```
PORT=$((1024+$RANDOM))
DIST_URL=tcp://0.0.0.0:$PORT
TRAIN_DATASET="custom_pascal_voc_trainval_2007"

EXP_NAME="experiments/voc2007_imagenet_init"
NUM_GPUS=2
python plain_train_net.py --dist-url $DIST_URL --config-file ./configs/faster_rcnn_R_50_C4_3x.yaml --num-gpus $NUM_GPUS  OUTPUT_DIR $EXP_NAME SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 8 DATALOADER.NUM_WORKERS 4 SEED 1235 DATASETS.TRAIN '("'$TRAIN_DATASET'",)' DATASETS.TEST '("custom_pascal_voc_test_2007",)' MODEL.ROI_HEADS.NUM_CLASSES 20 MODEL.WEIGHTS weights/R-50.pkl
```
This trains a detector on the PASCAL VOC 2007 set using a Res-50 backbone with Imagenet weights as initialization. This model achieves an AP50 of `72.5` AP50 on pascal voc 2007 test set
Weights for this model can be found in `experiments/voc2007_imagenet_init/model_final.pth`
Note that this model and training are just for completeness. Any system that uses imagenet weights as supervision will not be considered for the leaderboard.

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
This model achieves a AuC(@0.5) of `7.40%` on the COCO 2014 train set and discovers `19` objects with a CorLoc of `92.98`. For comparison a detector trained using supervised ImageNet weights discovers `13` objects with an AuC of `5.93` and a CorLoc of `90.28`.


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
