# 计算检测成功的比例
# 在两个阈值下  threshod =0.5 or 0.7
import datetime

import numpy as np
import random
from typing import List
import numpy as np
import io
import os
import json


RGBTmodel = 'fuseIRVIS'  # 'infrared' 'visible' 'fuseIRVIS' 'fuseIR_UnaffVIS'
# [0.5, 0.8, 0.05]
threshod = 0.8
H,W = 512,640

# 坐标形式是  x,y,w,h
def calIOU(test_position: List, gt_position: List) -> float:
    if not test_position or not gt_position:
        return 0.0
    xt1 = test_position[0]
    yt1 = test_position[1]
    xb1 = test_position[0] + test_position[2]
    yb1 = test_position[1] + test_position[3]
    xt2 = gt_position[0]
    yt2 = gt_position[1]
    xb2 = gt_position[0] + gt_position[2]
    yb2 = gt_position[1] + gt_position[3]
    # 确认交集左上角坐标
    # print(xt1, yt1, xb1, yb1)
    # print(xt2, yt2, xb2, yb2)

    xt, yt = max([xt1, xt2]), max([yt1, yt2])
    # 确认交集的右下角坐标
    xb, yb = min([xb1, xb2]), min([yb1, yb2])
    # print(xt, yt, xb, yb)
    if (xt > xb or yt > yb):
        return 0.0
    inter = (xb - xt) * (yb - yt)
    # print(xb-xt, yb-yt)
    union = test_position[2] * test_position[3] + gt_position[2] * gt_position[3] - inter
    if union == 0:
        return 0.0
    # print(inter,union)
    iou = inter / union
    # print("\n")
    return iou


def success_count(test: List, gt: List, threshod: float) -> int:
    success_count = 0
    assert len(test) == len(gt)
    for x, y in zip(test, gt):
        iou = calIOU(x, y)
        if iou > threshod:
            success_count += 1
            # print(iou)
    return success_count


def main():
    global sum_percentage


    # 指定数据集根目录
    if RGBTmodel == 'infrared' or RGBTmodel == 'visible':
        dataset_root = 'E:/PIPs/Anti-UAV-RGBT/test'
    else:
        dataset_root = 'E:/UNIFusion-main/outputs/Anti-UAV-RGBT/test'
    # 获取所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_file = f'{RGBTmodel}_results_{timestamp}.txt'  # 在文件名中包含时间戳

    # 读取分类数据
    with open('E:/PIPs/Anti-UAV-RGBT/label_new/categorized_data.json', 'r') as file:
        categorized_data = json.load(file)
    with open(output_file, 'w') as output:
        # print('subfolders',len(subfolders))
        all_acc = []
        alpha = 0.2
        beta = 0.3
        # 处理分类数据
        results = {}
        for category, subfolders in categorized_data.items():
            all_acc = []
            for subfolder_name in subfolders:
                subfolder = os.path.join(dataset_root, subfolder_name)  # 根据您的文件系统调整路径
                # 确保子文件夹路径存在
                if not os.path.exists(subfolder):
                    continue
                print("subfolder: ",subfolder)

                acc = 0.0

                test_path = subfolder+f"/{RGBTmodel}_bbox.json"
                gt_path = subfolder + f'/{RGBTmodel}.json'

                with open(test_path, 'r') as ftest:
                    data_test = json.load(ftest)
                    exist_test_values = data_test.get("exist", [])
                    bbox_rect = data_test.get("bbox_rect", [])

                with open(gt_path, 'r') as fgt:
                    data_gt = json.load(fgt)
                    exist_gt_values = data_gt.get("exist", [])
                    gt_rect = data_gt.get("gt_rect", [])

                    if RGBTmodel == 'visible':
                        y_resize = float(H / 1080)
                        x_resize = float(W / 1920)
                    else:
                        y_resize = float(H / 512)
                        x_resize = float(W / 640)

                    # print(len(bbox_rect),' ',len(gt_rect))
                    assert len(bbox_rect) == len(gt_rect)

                    # 计算视频中目标存在的总帧数 T
                    T = len(exist_gt_values)
                    T_exist = sum(1 for value in exist_gt_values if value == 1)

                    first_term = 0.0
                    second_term = 0.0
                    # 计算目标在视频中实际可见的帧数 T_exist
                    for idx,value in enumerate(exist_gt_values):

                        #  IoU_t is Intersection over Union (IoU) between the predicted tracking box and its corresponding ground-truth box,
                        #  p_t is the predicted visibility flag, it equals 1 when the predicted box is empty and 0 otherwise.
                        #  The v_t is the ground-truth visibility flag of the target, the indicator function δ(v_t>0) equals 1 when v_t > 0 and 0 otherwise.
                        v_t = exist_gt_values[idx]
                        p_t = 1 - exist_test_values[idx]

                        if len(gt_rect[idx])==4:
                            gt_rect[idx][0] = gt_rect[idx][0] * x_resize
                            gt_rect[idx][1] = gt_rect[idx][1] * y_resize
                            gt_rect[idx][2] = gt_rect[idx][2] * x_resize
                            gt_rect[idx][3] = gt_rect[idx][3] * y_resize

                        # print(bbox_rect)
                        iou_t = calIOU(bbox_rect[idx], gt_rect[idx])
                        # print('iou_t:',iou_t)
                        first_term += iou_t * v_t + p_t * (1-v_t)
                        if value == 1:
                            second_term += p_t * v_t

                acc = first_term / T - alpha*((second_term/T_exist)**beta)

                print("acc :",acc)

                all_acc.append(acc)


            # 计算每个分类的平均准确度并保存
            mean_acc = sum(all_acc) / len(all_acc) if all_acc else 0
            results[category] = mean_acc

        # 将平均准确度写入输出文件
        output.write(f"Mean acc: {mean_acc}\n")

        # 保存结果到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        output_file = f'categorized_results_{timestamp}.txt'
        with open(output_file, 'w') as output:
            for category, mean_acc in results.items():
                output.write(f"{category}: {mean_acc}\n")

        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
