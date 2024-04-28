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




import torch
from skimage.transform import resize


# 获取所有子文件夹的列表
subfolders = [f.path for f in os.scandir(output_root) if f.is_dir()]

# 遍历每个子文件夹
for subfolder in subfolders:
    tra_json_path_visible = os.path.join(subfolder, 'visible_tra.json')
    tra_json_path_infrared = os.path.join(subfolder, 'infrared_tra.json')

    # 加载可见光和红外光模态的轨迹
    with open(tra_json_path_visible, 'r') as f:
        json_data_visible = json.load(f)
    with open(tra_json_path_infrared, 'r') as f:
        json_data_infrared = json.load(f)

    # Access the 'trajs_e' key
    trajs_e_visible = np.array(json_data_visible['trajs_e'])
    trajs_e_infrared = np.array(json_data_infrared['trajs_e'])



    # Reshape trajs_e_visible
    trajs_e_visible = trajs_e_visible.reshape(-1, trajs_e_visible.shape[3], trajs_e_visible.shape[4])
    trajs_e_infrared = trajs_e_infrared.reshape(-1, trajs_e_infrared.shape[3], trajs_e_infrared.shape[4])
    print(f"Reshaped trajs_e_visible shape: {trajs_e_visible.shape}")
    print(f"Reshaped trajs_e_infrared shape: {trajs_e_infrared.shape}")

    src_points = []
    dst_points = []

    all_num=0
    for frame_visible, frame_infrared, is_valid_vis,is_valid_ir in zip(trajs_e_visible, trajs_e_infrared,json_data_visible['valid_tra'], json_data_infrared['valid_tra']):
        # print(f"frame_visible: {frame_visible}")
        # print(is_valid_vis)
        # print(is_valid_ir)
        num = 0
        all_num += 1
        for point_visible, point_infrared in zip(frame_visible, frame_infrared):
            # print(f"point_visible: {point_visible}")
            if is_valid_vis[num] and is_valid_ir[num]:
                src_points.append(np.array(point_visible))
                dst_points.append(np.array(point_infrared))
            num += 1

    print('Processing frame num:', all_num)
    # Convert the lists to NumPy arrays
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    # print(f"src_points shape: {src_points.shape}")
    # print(f"dst_points shape: {dst_points.shape}")
    # print(f"src_points: {src_points}")


    print(f"len src_points: {len(src_points)}")
    print(f"len dst_points: {len(dst_points)}")

    if len(src_points) >= 3:
        try:
            transform_matrix = cv2.estimateAffine2D(src_points, dst_points)[0]
        except cv2.error as e:
            print(f"Error in {subfolder}: {e}")
            continue
    else:
        print(f"Not enough points for transformation in {subfolder}")
        continue

    # 使用 os.path.split 分割路径
    head, tail = os.path.split(subfolder)
    vis_data_path = os.path.join(dataset_root, tail, 'visible')
    aff_vis_path = os.path.join(output_root, tail, 'affine_vis')

    # 创建 aff_vis_path 文件夹
    os.makedirs(aff_vis_path, exist_ok=True)

    # 获取所有 JPG 文件列表
    jpg_files = [f for f in os.listdir(vis_data_path) if f.lower().endswith('.jpg')]

    for jpg_file in jpg_files:
        # 加载可见光图像并resize
        visible_image = cv2.imread(os.path.join(vis_data_path, jpg_file))
        visible_image_resized = resize(visible_image, (H, W), anti_aliasing=True)
        visible_image_resized = (255 * visible_image_resized).astype('uint8')

        # 应用变换
        aligned_visible_image = cv2.warpAffine(visible_image_resized, transform_matrix, (W, H))

        # 保存或进一步处理aligned_visible_image
        aligned_image_path = os.path.join(aff_vis_path, jpg_file.replace('.jpg', '_aligned.jpg'))
        cv2.imwrite(aligned_image_path, aligned_visible_image)

        # print(f"Processed {jpg_file} done")

    print(f"Processed {subfolder} done")

