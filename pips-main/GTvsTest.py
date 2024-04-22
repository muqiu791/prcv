import numpy as np
import random
from typing import List
import numpy as np
import io
import os
import json
from PIL import Image
import cv2
import torch
import utils.improc
import imageio.v2 as imageio
import random
import glob
import matplotlib.pyplot as plt

import seaborn as sns

RGBTmodel = 'fuseIRVIS'  # 'infrared' 'visible' 'fuseIRVIS' 'fuseIR_UnaffVIS'

def draw_boxes(image, box, color=(0, 255, 0), thickness=2):
    """在图像上绘制目标框"""
    if len(box) == 0 or box == [0, 0, 0, 0]:
        return

    x, y, w, h = map(int, box)  # 将坐标值转换为整数
    print(f"Box values: x={x}, y={y}, w={w}, h={h}")  # 用于调试的打印语句

    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)



def save_image_with_boxes(image, output_folder, filename):
    """保存带有目标框的图像"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)

def main():

    # 指定数据集根目录
    if RGBTmodel == 'infrared' or RGBTmodel == 'visible':
        dataset_root = 'E:/PIPs/Anti-UAV-RGBT/train'
    else:
        dataset_root = 'E:/UNIFusion-main/outputs/Anti-UAV-RGBT/train'
    # 获取所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]

    for subfolder in subfolders:
        print("subfolder: ",subfolder)


        test_path = subfolder+'/fuseIRVIS_bbox.json'
        gt_path = subfolder + '/fuseIRVIS.json'

        with open(test_path, 'r') as ftest:
            data_test = json.load(ftest)
            exist_test_values = data_test.get("exist", [])
            bbox_rect = data_test.get("bbox_rect", [])

        with open(gt_path, 'r') as fgt:
            data_gt = json.load(fgt)
            exist_gt_values = data_gt.get("exist", [])
            gt_rect = data_gt.get("gt_rect", [])

        subsubfolder = os.path.join(subfolder, RGBTmodel)
        print('folr_path', subsubfolder)

        filenames = glob.glob(os.path.join(subsubfolder, '*.jpg'))

        output_folder = subfolder + '/' + f'{RGBTmodel}_with_boxes'
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)

        for idx, filename in enumerate(filenames):
            image = cv2.imread(filename)
            output_filename = os.path.splitext(os.path.basename(filename))[0] + '_with_boxes.jpg'
            output_path = os.path.join(output_folder, output_filename)

            print('11',output_path)
            # 绘制真实情况的目标框（绿色）
            if exist_gt_values:
                draw_boxes(image, gt_rect[idx], color=(0, 255, 0))

            # 绘制预测的目标框（红色）
            if exist_test_values:
                draw_boxes(image, bbox_rect[idx], color=(0, 0, 255))

            # 保存图像
            save_image_with_boxes(image, output_folder, output_filename)


if __name__ == "__main__":
    main()
