import datetime
import matplotlib.pyplot as plt
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


def main(alpha_value: float):
    global sum_percentage

    current_dir = os.path.dirname(__file__)
    # 指定数据集根目录
    if RGBTmodel == 'infrared' or RGBTmodel == 'visible':
        dataset_root = '../Anti-UAV-RGBT/test'
    else:
        dataset_root = '../fusing/outputs/Anti-UAV-RGBT/test'

    dataset_root = os.path.abspath(os.path.join(current_dir, dataset_root))
    
    # 获取所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]

    # print('subfolders',len(subfolders))
    all_acc = []
    alpha = alpha_value
    beta = 0.3
    for subfolder in subfolders:
        # print("subfolder: ",subfolder)
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

        all_acc.append(acc)

    mean_acc = sum(all_acc)/len(all_acc)
    print("mean acc:",mean_acc)

    return mean_acc

def write_results_to_file(alphas, mean_accs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_file = f'alpha_{RGBTmodel}_results_{timestamp}.txt'
    with open(output_file, 'w') as file:
        for alpha, mean_acc in zip(alphas, mean_accs):
            file.write(f"Alpha: {alpha}, Mean Accuracy: {mean_acc}\n")
    print(f"Results saved to {output_file}")

# 新增绘制函数
def plot_results(alphas, mean_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, mean_accs, marker='o', linestyle='-')
    plt.title('Mean Accuracy vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_alpha.png')
    plt.show()

if __name__ == "__main__":
    alphas = np.arange(0, 1.1, 0.1)  # 从0到1，步长为0.1
    mean_accs = []

    for alpha in alphas:
        mean_acc = main(alpha)
        mean_accs.append(mean_acc)

    plot_results(alphas, mean_accs)
    write_results_to_file(alphas, mean_accs)  # 将结果写入文件