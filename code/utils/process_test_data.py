# -*- coding: utf-8 -*- 
# @Time : 2021/4/8 15:12 
# @Author : aurorazeng
# @File : process_test_data.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

import os
import numpy as np
from PIL import Image
import math

root_path = r'/mnt/group-ai-medical-2/private/aurorazeng/data/multi_class_region/test'
image_dir = 'images'
mask_dir = 'masks'
con_image_dir = 'cov_image'
con_mask_dir = 'cov_mask'

tumor = [1, 19, 20]
lymph = [3, 10, 11, 14]
stroma = [2]
necrosis = [4]
others = [5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 18, 21]


def convertMultiLabels(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')
    for i in tumor:
        new_mask[mask == i] = 1
    # 2 - stroma - blue
    for i in stroma:
        new_mask[mask == i] = 2
    # 3 - lymphocytic_infiltrate - green
    for i in lymph:
        new_mask[mask == i] = 3
    # 4 - necrosis_or_debris - cyan
    for i in necrosis:
        new_mask[mask == i] = 4
    # 5 - others - yellow
    for i in others:
        new_mask[mask == i] = 5
    # 0: outside_roi - white
    new_mask[mask == 0] = 0
    return new_mask


image_dir = os.path.join(root_path, image_dir)
mask_dir = os.path.join(root_path, mask_dir)
con_image_dir = os.path.join(root_path, con_image_dir)
con_mask_dir = os.path.join(root_path, con_mask_dir)
if not os.path.isdir(con_image_dir):
    os.makedirs(con_image_dir)
if not os.path.isdir(con_mask_dir):
    os.makedirs(con_mask_dir)
for file in os.listdir(mask_dir):
    image = Image.open(os.path.join(image_dir, file))
    mask = Image.open(os.path.join(mask_dir, file))
    w = int(math.ceil(image.size[0] / 4 / 32) * 32)
    h = int(math.ceil(image.size[1] / 4 / 32) * 32)
    image = image.resize((w, h), Image.ANTIALIAS)
    mask = mask.resize((w, h), Image.NEAREST)
    mask = convertMultiLabels(np.asarray(mask))
    mask = Image.fromarray(mask)
    file = file.replace('2300', '920')
    image.save(os.path.join(con_image_dir, file))
    mask.save(os.path.join(con_mask_dir, file))