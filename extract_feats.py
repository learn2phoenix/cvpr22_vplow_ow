# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import pdb
import pickle

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

import vision_transformer as vits



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--data_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument('--box_file', default='.')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
    if not os.path.isdir(args.data_path):
        raise Exception('Data not found')
    # open image
    else:
        # img_files = random.sample(os.listdir(args.image_path), 20)
        img_files = os.listdir(args.data_path)
        if not os.path.isfile(args.box_file):
            raise Exception('Box file not valid')
        with open(args.box_file,'rb') as f:
            boxes = pickle.load(f)

        # pdb.set_trace()
        imgs = []
        for p in img_files:
            kk = int(p.lstrip("0").split('.')[0])
            if kk not in boxes:
                continue
            imgs.append(os.path.join(args.data_path, p))

    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    feat_dict = {}

    for id_, img_path in tqdm(enumerate(imgs), total=len(imgs)):
        imgs = []
        img_name = os.path.basename(img_path)
        key = int(img_name.lstrip("0").split('.')[0])

        with open(os.path.join(img_path), 'rb') as f:
            img_boxes = boxes[key]
            if not isinstance(img_boxes, list):
                img_boxes = [img_boxes]
            im = Image.open(f)
            for idx, b in enumerate(img_boxes):
                crop = im.crop(b.tolist())
                imgs.append(crop.convert('RGB'))
                """im_name = os.path.basename(img_path).split('.')[0]
                try:
                    crop.save('box_viz/'+im_name+'_{}.jpg'.format(idx))
                except:
                    pdb.set_trace()"""
        # imgs.append(im.convert('RGB'))
        if imgs:
            imgs = torch.stack([transform(img) for img in imgs])
            # pdb.set_trace()
            # make the image divisible by the patch size
            w, h = imgs.shape[1] - imgs.shape[1] % args.patch_size, imgs.shape[2] - imgs.shape[2] % args.patch_size
            img = imgs[:, :w, :h].unsqueeze(0)

            w_featmap = imgs.shape[-2] // args.patch_size
            h_featmap = imgs.shape[-1] // args.patch_size
            # pdb.set_trace()
            # attentions = model.get_last_selfattention(img.to(device))

            features = model(imgs.to(device)).cpu().numpy()
            feats = [features[l] for l in range(len(features))]
        else:
            feats = []
        # pdb.set_trace()
        feat_dict[key] = feats
    with open(os.path.join(args.output_dir, 'feats_dino.pkl'), 'wb') as f:
        pickle.dump(feat_dict, f)
