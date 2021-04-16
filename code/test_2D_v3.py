# -*- coding: utf-8 -*- 
# @Time : 2021/4/16 15:10 
# @Author : aurorazeng
# @File : test_2D_v3.py 
# @license: (C) Copyright 2021-2026, aurorazeng; No reprobaiction without permission.


import argparse
import os
import shutil
import math

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric

# from networks import linkNetBase
from networks.net_factory import net_factory

import os
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../csv_data', help='path of data')
parser.add_argument('--exp', type=str,
                    default='/Fu_LinkNetWithdrop_4_8_1715_labeled', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='LinkNetBaseWithDrop', help='model_name')
parser.add_argument('--num_classes', type=int, default=6,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--stride', type=int, default=2048, help='patch stride for WSI')
parser.add_argument('--target_mpp', type=float, default=0.92, help='the train model mpp')
parser.add_argument('--pad', type=int, default=128, help='the pad for the patch size')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if (pred.sum() > 0 and gt.sum() > 0):
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    elif gt.sum() == 0:
        return -1, -1, -1
    else:
        return 0, 500, 500


def test_single_volume(case_path, net, stride, pad, target_mpp):
    image_path = case_path.split(',')[0]
    mask_path = case_path.split(',')[1]
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # get test image mpp
    name = os.path.basename(image_path).split('/')[-1]
    name_noext, _ = os.path.splitext(name)
    print(name_noext)
    mpp = float(name_noext.split('MPP-')[1])

    # test images mpp != model mpp, resize image
    scale = round(mpp / target_mpp, 4)
    resize_width = int(math.ceil(image.size[0] * scale / 32) * 32)
    resize_height = int(math.ceil(image.size[1] * scale / 32) * 32)
    ds_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)
    ds_mask = mask.resize((resize_width, resize_height), Image.NEAREST)

    # convert  ds_image ds_mask to ndarry
    ds_image = np.asarray(ds_image, np.float32)
    ds_label = np.asarray(ds_mask, np.float32)
    ds_label = ds_label.astype(np.int)
    ds_label = convertMultiLabels(ds_label)
    ds_image /= 255

    # extract patches and prediction
    patch_size = stride + 2 * pad
    max_y = int((np.ceil(resize_height / stride) - 1) * stride) + patch_size
    max_x = int((np.ceil(resize_width / stride) - 1) * stride) + patch_size

    ds_prediction = np.zeros((max_y, max_x), dtype='uint8')

    image = np.zeros((max_y, max_x, 3), dtype='uint8')
    image[pad:ds_image.shape[0] + pad, pad:ds_image.shape[1] + pad, :] = ds_image

    label = np.zeros((max_y, max_x), dtype='uint8')
    label[pad:ds_mask.shape[0] + pad, pad:ds_mask.shape[1] + pad] = ds_label

    y_list = range(0, max_y, stride)
    x_list = range(0, max_x, stride)

    for y in y_list:
        for x in x_list:
            patch = image[y:y + patch_size, x:x + patch_size, :]
            label = label[y:y + patch_size, x:x + patch_size]

            patch = np.transpose(patch, (2, 0, 1))
            input = torch.from_numpy(patch).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_main = net(input)
                out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                prediction = out
            ds_prediction[y:y + stride, x:x + stride] = prediction[pad:stride + pad, pad:stride + pad]

    overlay_mask = np.zeros((ds_label.shape[0], ds_label.shape[1]), dtype='uint8')
    overlay_mask[ds_label > 0] = 1
    ds_prediction = ds_prediction[0:ds_label.shape[0], ds_label.shape[1]]
    assert overlay_mask.shape == ds_prediction.shape, 'Error'
    ds_prediction = ds_prediction * overlay_mask
    metric_list = []
    for i in range(1, 6):
        metric_list.append(calculate_metric_percase(ds_prediction == i, ds_label == i))
    return metric_list


def Inference(FLAGS):
    with open(FLAGS.root_path + '/val_3_25.txt', 'r') as f:
        case_list = f.readlines()
    case_list = [item.replace('\n', '')
                 for item in case_list]
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    file_list = os.listdir(snapshot_path)
    filedir = []
    for i in file_list:
        if i.__contains__('pth') and int(i.split('.')[0].split('_')[1]) % 2000 == 0:
            filedir.append(i)
    filedir.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    target_txt = 'dice.txt'
    f = open(test_save_path + target_txt, 'a')
    for file in filedir:
        if file.__contains__('pth'):
            save_mode_path = os.path.join(
                snapshot_path, file)
            net.cuda()
            net.load_state_dict(torch.load(save_mode_path))
            print("init weight from {}".format(save_mode_path))
            net.eval()

            list = []
            metric = []
            for case in tqdm(case_list):
                metric_i = test_single_volume(
                    case, net, stride=args.stride, pad=args.pad, target_mpp=args.target_mpp)
                list.append(metric_i)
            list = np.asarray(list)
            metric_list = []
            for i in range(0, 5):
                metric_list.append(list[:, i])
            metric_list = np.asarray(metric_list)
            for i in range(0, 5):
                num = 0
                metric_i = 0
                for j in range(0, metric_list.shape[1]):
                    if np.sum(metric_list[i][j]) > 0:
                        num += 1
                        metric_i += metric_list[i][j]
                metric_i = metric_i / num
                metric.append(metric_i)
            avg_metric = np.asarray(metric)
            line = file
            for i in range(0, 5):
                line = line + ',' + str(round(avg_metric[i][0], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[0], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[1], 4))
            line = line + ',' + str(round(np.mean(avg_metric, axis=0)[2], 4))
            line = line + '\n'
            f.write(line)
    f.close()


if __name__ == "__main__":
    Inference()
