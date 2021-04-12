# -*- coding: utf-8 -*- 
# @Time : 2021/4/8 15:10 
# @Author : aurorazeng
# @File : compute_weight.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.

import os
import numpy as np
import cv2

if __name__ == "__main__":

    # label_path = r'E:\CrowdsourcingDataset-Amgadetal2019\dataset\train\patch_0.92\labels'
    label_path = r'C:\Users\aurorazeng\Desktop\data\train\patch_0.92\labels'
    nameList = os.listdir(label_path)

    total = np.array(0, dtype=np.int64)
    tumor = np.array(0, dtype=np.int64)
    stroma = np.array(0, dtype=np.int64)
    lymph = np.array(0, dtype=np.int64)
    necrosis = np.array(0, dtype=np.int64)
    others = np.array(0, dtype=np.int64)

    for name in nameList:
        print(name)
        name_noext, _ = os.path.splitext(name)
        label = cv2.imread(os.path.join(label_path, name), 0)


        tumor += np.sum(label == 1)
        stroma += np.sum(label == 2)
        lymph += np.sum(label == 3)
        necrosis += np.sum(label == 4)
        others += np.sum(label == 5)
        total += label.shape[0] * label.shape[1] - np.sum(label == 0)

        if total < 0:
            break

    print('total: %d' % total)
    print('tumor: %d' % tumor)
    print('stroma: %d' % stroma)
    print('lymph: %d' % lymph)
    print('necrosis: %d' % necrosis)
    print('others: %d' % others)

    w_tumor = 1 - tumor / total
    w_stroma = 1 - stroma / total
    w_lymph = 1 - lymph / total
    w_necrosis = 1 - necrosis / total
    w_others = 1 - others / total
    print(w_tumor, w_stroma, w_lymph, w_necrosis, w_others)