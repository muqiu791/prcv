import time
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

# trajs_e shape torch.Size([B, S, N, 2])

RGBTmodel = 'infrared'  # 'infrared' 'visible'

H, W = 512, 640

# 指定数据集根目录
dataset_root = '../../Anti-UAV-RGBT/test'
output_root = '../../tracking/Myoutput/Anti-UAV-RGBT/test'

# 获取所有子文件夹的列表
subfolders = [f.path for f in os.scandir(output_root) if f.is_dir()]

# Initialize an empty list to store all distances
all_distances = []

for subfolder in subfolders:
    # 拼接 'infrared.json' 文件路径
    tra_json_path = os.path.join(subfolder, f'{RGBTmodel}_tra.json')
    print('folr_path', tra_json_path)

    # 使用 os.path.split 分割路径
    head, tail = os.path.split(subfolder)
    gt_json_path = os.path.join(dataset_root, tail, f'{RGBTmodel}.json')
    print('gt_json_path:', gt_json_path)

    # 初始化一个存储轨迹坐标数据的列表
    all_trajs_e = []

    read_start_time = time.time()

    with open(gt_json_path, 'r') as gt_json_file:
        data_gt = json.load(gt_json_file)

        exist_gt_values = data_gt.get("exist", [])

        gt_values = data_gt.get("gt_rect", [])

    with open(tra_json_path, 'r') as tra_json_file:
        data_tra = json.load(tra_json_file)

        # 获取 "exist" 键的值，如果键不存在，则返回一个默认的空列表
        exist_tra_values = data_tra.get("exist", [])
        print('exist_tra_values:', len(exist_tra_values))

        trajs_e_values = data_tra.get("trajs_e", [])

        # if you want to convert it to a PyTorch tensor
        trajs_e_tensor = torch.tensor(trajs_e_values)
        print('trajs_e_tensor shape:', trajs_e_tensor.shape)
        _, B, S, N, _ = trajs_e_tensor.shape

        # 将第一维和第二维合并为一个新的第一维，得到形状 [, N, 2]
        reshaped_trajs_e_tensor_values = trajs_e_tensor.view(-1, N, 2)
        print('reshaped_trajs_e_tensor shape:', reshaped_trajs_e_tensor_values.shape)

        valid_list = []
        distances_list = []

        # Initialize an empty list to store the binary list for the current iteration
        valid_list = []
        # Initialize an empty list to store the rectangle information
        rectangles_info = []
        bbox_exist = []
        # Initialize an empty list to store the ratios of distances between valid points
        distances_ratios_list = []

        # 遍历所有帧，计算距离差，判断是否有效
        for idx, exist_tra_value in enumerate(exist_tra_values):
            if exist_tra_value == 1 and exist_gt_values[idx] == 1:
                # 获取第 idx 个轨迹的追踪坐标
                tra_xy = reshaped_trajs_e_tensor_values[idx]

                gt_rect_value = gt_values[idx]

                if RGBTmodel == 'visible':
                    y_resize = float(H / 1080)
                    x_resize = float(W / 1920)
                else:
                    y_resize = float(H / 512)
                    x_resize = float(W / 640)

                gt_rect_value[0] = gt_rect_value[0] * x_resize
                gt_rect_value[1] = gt_rect_value[1] * y_resize
                gt_rect_value[2] = gt_rect_value[2] * x_resize
                gt_rect_value[3] = gt_rect_value[3] * y_resize

                left = int(gt_rect_value[0])
                top = int(gt_rect_value[1])
                right = int(gt_rect_value[0] + gt_rect_value[2])
                bottom = int(gt_rect_value[1] + gt_rect_value[3])

                N_ = 4

                # 计算每个轨迹点的真实坐标xy
                xy_coords = []
                relative_positions = [
                    [0, 0], [0.333333, 0], [0.666666, 0], [1, 0],
                    [0, 0.333333], [0.333333, 0.333333], [0.666666, 0.333333], [1, 0.333333],
                    [0, 0.666666], [0.333333, 0.666666], [0.666666, 0.666666], [1, 0.666666],
                    [0, 1], [0.333333, 1], [0.666666, 1], [1, 1],
                    [-0.2,-0.2],[1.2,-0.2],[-0.2,1.2],[1.2,1.2],[-0.4,-0.4],[1.4,-0.4],[-0.4,1.4],[1.4,1.4]
                ]
                for rel_pos in relative_positions:
                    # 计算实际坐标
                    actual_x = gt_rect_value[0] + rel_pos[0] * gt_rect_value[2]
                    actual_y = gt_rect_value[1] + rel_pos[1] * gt_rect_value[3]
                    xy_coords.append([actual_x, actual_y])

                # 将xy_coords转换为Tensor，以便进行后续的距离计算
                xy = torch.tensor(xy_coords, device=tra_xy.device)
                # print('xy shape:', xy.shape)
                # print('xy:', xy)
                # print('tra_xy:', tra_xy)

                xy = xy.unsqueeze(0)  # 将xy形状从[24, 2]扩展到[1, 24, 2]

                tra_xy = tra_xy.to(xy.device)

                # 计算欧氏距离
                distances = torch.norm(tra_xy - xy, dim=2)

                # 将 distances 转换为 Python 列表，并去除第一维度
                distances_list.append(distances.squeeze().tolist())

                # Check if distances are less than 4 and append binary values to the list
                binary_values = (distances.squeeze() <= 4).int().tolist()
                valid_list.append(binary_values)

            else:
                # 添加[1,20]维度的列表，值全部为 -1
                distances_list.append([-1] * N)
                valid_list.append([0] * N)

    print('distances_list:', len(distances_list))
    # 将列表转换为 JSON 字符串
    distances_json = json.dumps(distances_list)

    valid_data = {"valid_tra": valid_list}
    valid_json = json.dumps(valid_data)

    # 使用 os.path.split() 分割路径，获取最后两级子目录
    subfold_split = subfolder.split('/')
    output_subfolder = os.path.join(subfold_split[-2], subfold_split[-1])

    # 构造输出文件路径
    dist_json_path = './MyOutput/' + output_subfolder + '/' + RGBTmodel + '_distances.json'

    # 确保目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(dist_json_path), exist_ok=True)

    with open(dist_json_path, 'w') as output_file:
        output_file.write(distances_json)

    print(f'Distances saved to {dist_json_path}')

    # 将 valid_json 附加到 tra_json_path 文件
    # 以读取模式打开现有的JSON文件
    with open(tra_json_path, 'r') as file:
        # 加载现有的JSON内容
        existing_data = json.load(file)

    # 将新的"valid_data"字典追加到现有内容中
    existing_data.update(valid_data)

    # 以写入模式打开JSON文件
    with open(tra_json_path, 'w') as file:
        # 将更新后的JSON内容写回文件
        json.dump(existing_data, file, indent=2)

    read_time = time.time() - read_start_time
    iter_start_time = time.time()

    # 将当前迭代的距离附加到列表，并在附加之前展平为一维数组
    all_distances.extend(np.array(distances_list).ravel())

print('all_distances:', len(all_distances))
# 将所有距离转换为 NumPy 数组
# 假设你的数据在名为 all_distances_array 的 NumPy 数组中
all_distances_array = np.array(all_distances)

# 设置横轴范围为-1到20
plt.hist(all_distances_array, bins=50, range=(-1, 20), color='blue', alpha=0.7)
plt.title('Distance Distribution')
plt.xlabel('Distance')
plt.ylabel('Frequency')

# 在图上标注0到4范围的分布
plt.axvspan(0, 4, color='red', alpha=0.3)

# 计算并标注分布区间内的样本数
samples_in_range = ((all_distances_array >= 0) & (all_distances_array <= 4)).sum()
plt.text(2, 2000, f'Samples in Range [0-4]: {samples_in_range}', fontsize=10, color='red')

plt.show()
