# 无关键点 无背景校正

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

random.seed(42)
np.random.seed(42)
# trajs_e shape torch.Size([1, 8, 24, 2])

# RGBTmodel = 'visible'  # 'infrared' 'visible' 'fuseIRVIS' 'fuseIR_UnaffVIS'
for RGBTmodel in ['fuseIRVIS']:
    H,W = 512, 640

    # 指定数据集根目录
    if RGBTmodel == 'infrared' or RGBTmodel == 'visible':
        dataset_root = 'E:/PIPs/Anti-UAV-RGBT/test'
    else:
        dataset_root = 'E:/UNIFusion-main/outputs/Anti-UAV-RGBT/test'
    output_root = 'E:/PIPs/pips-main/Myoutput/Anti-UAV-RGBT/test'

    # 获取所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(output_root) if f.is_dir()]

    # Initialize an empty list to store all distances
    all_distances = []

    for subfolder in subfolders:
        # 拼接 'infrared.json' 文件路径
        tra_json_path = os.path.join(subfolder, f'{RGBTmodel}_tra_wokey.json')
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

            # 将第一维和第二维合并为一个新的第一维，得到形状 [n*50, 20, 2]
            reshaped_trajs_e_tensor_values = trajs_e_tensor.view(-1, N, 2)
            print('reshaped_trajs_e_tensor shape:', reshaped_trajs_e_tensor_values.shape)

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
                    tra_xy=reshaped_trajs_e_tensor_values[idx]

                    gt_rect_value = gt_values[idx]

                    # N个追踪点的gt坐标,N = N_*N_ + M
                    N_ = 4
                    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')

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

                    grid_y = (gt_rect_value[1] + grid_y.reshape(B, -1) / float(N_ - 1) * gt_rect_value[3]) * y_resize
                    grid_x = (gt_rect_value[0] + grid_x.reshape(B, -1) / float(N_ - 1) * gt_rect_value[2]) * x_resize

                    # 计算每个点在目标框中的相对位置
                    relative_positions = torch.stack([(grid_x - gt_rect_value[0]) / gt_rect_value[2],
                                                      (grid_y - gt_rect_value[1]) / gt_rect_value[3]], dim=-1)
                    print('relative_positions:', relative_positions)
                    # 计算目标框的四个顶点
                    left = int(gt_rect_value[0])
                    top = int(gt_rect_value[1])
                    right = int(gt_rect_value[0] + gt_rect_value[2])
                    bottom = int(gt_rect_value[1] + gt_rect_value[3])

                    M = N - N_ ** 2

                    # 计算对角线长度
                    diagonal_length = torch.sqrt(torch.tensor((gt_rect_value[2] ** 2 + gt_rect_value[3] ** 2)).float())
                    # 设置固定距离，这里您已经定义为对角线长度的10%
                    fixed_distance = 0.1 * diagonal_length

                    # 计算对角线方向
                    diagonal_direction = torch.tensor(
                        [gt_rect_value[2] / diagonal_length, gt_rect_value[3] / diagonal_length],
                        device='cuda')

                    # 在每个顶点外侧放置一个额外的点
                    for i in range(M):
                        # 每一组4个点后，距离加倍
                        if i % 4 == 0:
                            fixed_distance *= 2
                        # 根据顶点的位置和对角线方向计算额外点的坐标
                        x = torch.ones((1, 1), device=torch.device('cuda')) \
                            * (left - diagonal_direction[0] * fixed_distance if (i % 4) % 2 == 0 else right +diagonal_direction[0] * fixed_distance)
                        y = torch.ones((1, 1), device=torch.device('cuda')) \
                            * (bottom + diagonal_direction[1] * fixed_distance if (i % 4) > 1 else top -diagonal_direction[1] * fixed_distance)

                        grid_y = torch.cat([grid_y, y], dim=1)
                        grid_x = torch.cat([grid_x, x], dim=1)

                    xy = torch.stack([grid_x, grid_y], dim=-1)  # B, N*N, 2
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
                    binary_values = [0] * N
                    valid_list.append(binary_values)

                # 初始化一个空列表来存储坐标和索引的字典
                valid_points_with_indices = []

                # 遍历每个点的二值化值
                for point_idx, is_valid in enumerate(binary_values):
                    if is_valid and point_idx <= 15:
                        # 提取有效点的坐标
                        point_coordinates = tra_xy[point_idx]

                        # 创建一个字典来存储坐标和索引
                        point_info = {
                            'coordinates': point_coordinates,
                            'index': point_idx
                        }

                        # 将字典添加到列表中
                        valid_points_with_indices.append(point_info)

                # If there are enough valid points in the frame
                if len(valid_points_with_indices) >= 3:
                    bbox_exist.append(1)

                    H_list = []
                    W_list = []
                    # 遍历列表中的点信息，计算每一对点之间的坐标差
                    for i in range(len(valid_points_with_indices)):
                        for j in range(i + 1, len(valid_points_with_indices)):  # 从i+1开始以避免重复计算
                            # 获取两个点的坐标
                            point_i_coordinates = valid_points_with_indices[i]['coordinates']
                            point_j_coordinates = valid_points_with_indices[j]['coordinates']

                            point_i_index = valid_points_with_indices[i]['index']
                            point_j_index = valid_points_with_indices[j]['index']

                            point_i_relative_position = relative_positions[0, point_i_index, :]
                            point_j_relative_position = relative_positions[0, point_j_index, :]

                            # 计算两个点在每个维度上的差值
                            diffs0 = [abs(a - b) for a, b in zip(point_i_coordinates, point_j_coordinates)]
                            diffs1 = [abs(a - b) for a, b in zip(point_i_relative_position, point_j_relative_position)]


                            if diffs0[0] and diffs1[0]:
                                W_list.append(diffs0[0] / diffs1[0])
                            if diffs0[1] and diffs1[1]:
                                H_list.append(diffs0[1] / diffs1[1])

                    # print('W_list:', W_list)
                    # print('H_list:', H_list)

                    W_list_cpu = [w.item() for w in W_list]  # 使用 .item() 将张量转换为Python数值
                    H_list_cpu = [h.item() for h in H_list]

                    # 现在 W_list_cpu 和 H_list_cpu 是包含Python数值的列表，可以在CPU上使用
                    W_list_cpu.sort()
                    H_list_cpu.sort()
                    width = np.median(W_list_cpu) if W_list_cpu else last_valid_rect[2]
                    height = np.median(H_list_cpu) if H_list_cpu else last_valid_rect[3]

                    topleft_x_list = []
                    topleft_y_list = []
                    # 遍历列表中的点信息，计算每一对点之间的坐标差
                    for i in range(len(valid_points_with_indices)):
                        # 获取两个点的坐标
                        point_i_coordinates = valid_points_with_indices[i]['coordinates']
                        point_i_index = valid_points_with_indices[i]['index']
                        point_i_relative_position = relative_positions[0, point_i_index, :]

                        # # 估算目标框的坐标
                        topleft_x = point_i_coordinates[0] - point_i_relative_position[0] * width
                        topleft_y = point_i_coordinates[1] - point_i_relative_position[1] * height

                        topleft_x_list.append(topleft_x)
                        topleft_y_list.append(topleft_y)

                    x_list_cpu = [x.item() for x in topleft_x_list]  # 使用 .item() 将张量转换为Python数值
                    y_list_cpu = [y.item() for y in topleft_y_list]

                    x_list_cpu.sort()
                    y_list_cpu.sort()
                    x = np.median(x_list_cpu) if x_list_cpu else last_valid_rect[0]
                    y = np.median(y_list_cpu) if y_list_cpu else last_valid_rect[1]

                    # width = round(width)
                    # height = round(height)
                    # x = round(x)
                    # y = round(y)
                    # print('adding bbox:',x, y, width, height)
                    # Append the rectangle information to the list
                    rectangles_info.append((x, y, width, height))
                    last_valid_rect = (x, y, width, height)
                else:
                    # If there are not enough valid points in the frame
                    # print('not exist')
                    bbox_exist.append(0)
                    rectangles_info.append([])

        # 使用 os.path.split() 分割路径，获取最后两级子目录
        subfold_split = subfolder.split('/')
        output_subfolder = os.path.join(subfold_split[-2], subfold_split[-1])
        head, tail = os.path.split(subfolder)

        # print('bbox_exist:',bbox_exist)
        # # Print the rectangles information
        # print('rectangles_info:',rectangles_info)

        bbox_data = {"exist": bbox_exist, "bbox_rect": rectangles_info}
        bbox_json = json.dumps(bbox_data,indent=2)

        # 保存目标框文件
        bbox_json_path = dataset_root + '/' + tail + '/' + RGBTmodel + '_bbox_8_0.json'
        # 确保目录存在，如果不存在则创建
        print('bbox_o=path',bbox_json_path)
        os.makedirs(os.path.dirname(bbox_json_path), exist_ok=True)
        with open(bbox_json_path, 'w') as output_file:
            output_file.write(bbox_json)
        print(f'bbox saved to {bbox_json_path}')


        print('distances_list:', len(distances_list))
        # 将列表转换为 JSON 字符串
        distances_json = json.dumps(distances_list)

        valid_data = {"valid_tra": valid_list}
        valid_json = json.dumps(valid_data)

        # 保存距离数据文件
        dist_json_path = './MyOutput/' + output_subfolder + '/' + RGBTmodel + '_distances_8_0.json'
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
    plt.savefig(f'{RGBTmodel}_distances.png')