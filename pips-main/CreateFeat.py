import os

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载预训练的ResNet模型
model = models.resnet18(pretrained=True)
model.eval()  # 设置为评估模式

# 要获取的层名称和它们的输出
layers = ['layer1', 'layer2', 'layer3', 'layer4']
outputs = []

# 修改模型以保存指定层的输出
for layer_name in layers:
    layer = getattr(model, layer_name)
    def copy_data(m, i, o): outputs.append(o)
    layer.register_forward_hook(copy_data)

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并预处理图像
img = Image.open("E:\\UNIFusion-main\\outputs\\Anti-UAV-RGBT\\test\\20190925_124000_1_1\\fuseIRVIS\\fuseIRVIS0000.jpg")  # 更改为你的图片路径
img_tensor = preprocess(img).unsqueeze(0)  # 添加一个批次维度

# 使用模型获取特征图
with torch.no_grad():
    model(img_tensor)

# 为保存的特征图创建一个目录
output_dir = "feature_maps"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 可视化特征图并保存
for index, layer_output in enumerate(outputs):
    feature_map = layer_output[0, 0].cpu().data  # 获取每个层输出的第一个特征图
    plt.figure()
    plt.imshow(feature_map, cmap='gray')
    plt.axis('off')
    # 去除周围的白边
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # 构建每个特征图的保存路径
    save_path = os.path.join(output_dir, f"feature_map_layer_{index+1}.png")
    # 保存特征图到文件，去除周围的空白边距
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像，以避免内存中打开太多图像
# import torch
# import torch.nn.functional as F
#
# # 假设指定点的二维坐标为 (x, y)，这里使用图像中心点作为例子
# # 注意：这个坐标是基于原始图像的尺寸
# original_width, original_height = img.size
# x, y = 350, 280  # 指定的点坐标
#
# # 计算映射到第一层特征图上的坐标
# feature_map = outputs[0]  # 第一层的特征图输出
# B, C, H, W = feature_map.size()  # 获取特征图的尺寸
#
# # 预处理后的图像尺寸
# target_height, target_width = 224, 224  # 根据preprocess中的transforms.CenterCrop决定
#
# # 将指定点的坐标映射到预处理后的图像尺寸
# mapped_x = x / original_width * target_width
# mapped_y = y / original_height * target_height
#
# # 归一化坐标到 [-1, 1]，以适应grid_sample的需要
# normalized_x = (mapped_x / (target_width - 1)) * 2 - 1
# normalized_y = (mapped_y / (target_height - 1)) * 2 - 1
#
# # 创建采样grid
# grid = torch.tensor([[[[normalized_y, normalized_x]]]], dtype=torch.float)
#
# # 执行双线性采样
# sampled_feature_vector = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=True)
#
# # 去除不必要的维度，以便与原始特征图进行点乘
# sampled_feature_vector = sampled_feature_vector[:, :, 0, 0]  # [B, C, 1, 1] -> [B, C]
#
# # 将采样向量扩展为与特征图相同的尺寸并执行点乘操作
# expanded_vector = sampled_feature_vector.unsqueeze(-1).unsqueeze(-1)  # [B, C] -> [B, C, 1, 1]
# result_feature_map = feature_map * expanded_vector  # 点乘
#
# # 选择一个通道进行可视化
# channel_to_visualize = 0
# plt.figure()
# plt.imshow(result_feature_map[0, channel_to_visualize].cpu().detach().numpy(), cmap='gray')
# plt.axis('off')
# plt.show()