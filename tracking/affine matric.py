import numpy as np
import cv2
import os
import json

import torch
from skimage.transform import resize

# 设置全局变量
RGBTmodel = 'infrared'  # 'infrared' 'visible'
# B = 1
# S = 8
# N = 16 + 8  # number of points to track
H, W = 512, 640
dataset_root = 'E:/PIPs/Anti-UAV-RGBT/test'
output_root = 'E:/PIPs/pips-main/Myoutput/Anti-UAV-RGBT/test'

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

