#!/usr/bin/env python
# coding: utf-8

# ### Convert Udacity Self-driving Car object-detection dataset annotation into coco format
# URL : https://github.com/udacity/self-driving-car

# imports
import argparse
import datetime
import json
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pprint import pprint

# setup constant
IMG_DIR = "object-detection-crowdai"
anno_coco_file = "labels_crowdai_coco.json"
# Read existing csv as dataframe
df_anno = pd.read_csv("labels_crowdai.csv")
print(df_anno.shape)
df_anno.head()

# Functions
## COCO specific functions
def gen_coco_cat_d(df_anno):
    coco_cat_l = []
    for cat_id, cat_name in enumerate(df_anno.Label.unique()):
        cat_dict = dict(id=cat_id+1, name=cat_name, supercategory='none')
        coco_cat_l.append(cat_dict)
    return coco_cat_l

def get_cat_id(cat_name, coco_cat_l):
    cat_id = None
    for cat_d in coco_cat_l:
        if cat_d['name'] == cat_name:
            cat_id = cat_d['id']
            break
    return cat_id

def gen_coco_images_d(df_anno, img_dir):
    coco_img_l = []
    for img_id, img_name in enumerate(df_anno.Frame.unique()):
        filepath = os.path.join(img_dir, img_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError
        img_dict = dict(id=int(os.path.splitext(img_name)[0]), file_name=img_name, height=1200, width=1920)
        coco_img_l.append(img_dict)
    print("Number of images - {}".format(len(coco_img_l)))
    return coco_img_l

def get_img_id(img_name, img_dir):
    filepath = os.path.join(img_dir, img_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError
    img_id = int(os.path.splitext(img_name)[0])
    return img_id

def gen_coco_anno_d(df_anno, img_dir, coco_cat_l):
    coco_anno_l = []
    for index, row in df_anno.iterrows():
        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax
        w = xmax - xmin
        h = ymax - ymin
        bbox = [xmin, ymin, w, h]
        img_name = row.Frame
        img_id = get_img_id(img_name, img_dir)
        cat_name = row.Label
        cat_id = get_cat_id(cat_name, coco_cat_l)

        anno_dict = dict(id=index+1, image_id=img_id, category_id=cat_id, bbox=bbox)
        coco_anno_l.append(anno_dict)
    print("Number of Annotations - {}".format(len(coco_anno_l)))
    return coco_anno_l

# main executable part
if __name__ == "__main__":
    # Generate list of categories
    COCO_CAT_L = gen_coco_cat_d(df_anno)
    pprint(COCO_CAT_L)

    # Generate list of images
    COCO_IMG_L = gen_coco_images_d(df_anno, IMG_DIR)
    pprint(COCO_IMG_L[:5])

    # Generate list of annotations
    COCO_ANNO_L = gen_coco_anno_d(df_anno, IMG_DIR, COCO_CAT_L)
    pprint(COCO_ANNO_L[:5])

    # Create final COCO-styled annotation dictionary
    COCO_ANNO_DICT = dict(
        categories = COCO_CAT_L,
        images = COCO_IMG_L,
        annotations = COCO_ANNO_L,
    #     type = "instances"
    )
    print(COCO_ANNO_DICT.keys())

    # dump to json file
    with open(anno_coco_file, 'w') as fp:
        json.dump(COCO_ANNO_DICT, fp)

