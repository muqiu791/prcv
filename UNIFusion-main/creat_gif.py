import os
from PIL import Image

def create_gif(folder_path, output_gif_path, num_images):
    # 获取文件夹中的所有文件
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 仅选择指定数量的图像
    files = files[:num_images]

    images = []

    # 读取图像并添加到列表中
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = Image.open(image_path)
        images.append(img)

    # 保存为 GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f'GIF saved to {output_gif_path}')

if __name__ == "__main__":
    # 指定文件夹路径、输出 GIF 路径和图像数量
    folder_path = "E:\\UNIFusion-main\\outputs\\Anti-UAV-RGBT\\train\\20190925_131530_1_2\\fuseIRVIS"
    num_images = 50  # 指定图像数量
    output_gif_path = folder_path + f'{num_images}.gif'

    create_gif(folder_path, output_gif_path, num_images)
