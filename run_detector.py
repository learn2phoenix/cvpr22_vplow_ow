import itertools
import logging
from collections import OrderedDict, defaultdict
import os
import json
import pickle

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, launch, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import print_csv_format, inference_on_dataset

from plain_train_net import Res5ROIHeadsExtraNorm

logger = logging.getLogger("detectron2")


def test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = DummyEvaluator(output_dir=cfg.OUTPUT_DIR)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


class DummyEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name='predictions', distributed=True, output_dir=None):
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            # Commenting for current version
            # if "proposals" in output:
            #     prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        predictions = list(itertools.chain(*[x["instances"] for x in predictions]))
        all_boxes = defaultdict(list)
        for pred in predictions:
            box = list(pred['bbox'])
            box[2] += box[0]
            box[3] += box[1]
            # saved boxes are in xyxy format
            all_boxes[pred['image_id']].append(np.asarray(box))

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()
            with open(os.path.join(self._output_dir, "preds.pkl"), "wb") as f:
                pickle.dump(all_boxes, f)
        self._results = OrderedDict()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def extract_predictions(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    test(cfg, model)
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        extract_predictions,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
