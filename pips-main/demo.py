import time
import numpy as np
import io
import os
import json
from PIL import Image
import cv2
from matplotlib import pyplot as plt

import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from skimage.transform import resize

random.seed(42)
np.random.seed(42)

RGBTmodel = 'fuseIR_UnaffVIS'  # 'infrared' 'visible' 'fuseIRVIS' 'fuseIR_UnaffVIS'

def run_model(model, rgbs, N, sw, subfolder, gt_rect_value):
    rgbs = rgbs.cuda().float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B * S, C, H, W)
    H_, W_ = 512, 640  # 512, 640  1080，1920
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)


    # pick N points to track; we'll use a uniform grid
    # N_ = np.sqrt(N).round().astype(np.int32)
    N_ = 4
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')

    if RGBTmodel == 'visible':
        y_resize = float(H / 1080)
        x_resize = float(W / 1920)
    else:
        y_resize = float(H / 512)
        x_resize = float(W / 640)
    grid_y = (gt_rect_value[1] + grid_y.reshape(B, -1) / float(N_ - 1) * gt_rect_value[3]) * y_resize
    grid_x = (gt_rect_value[0] + grid_x.reshape(B, -1) / float(N_ - 1) * gt_rect_value[2]) * x_resize

    M = N - N_**2
    #设gt_rect_value是一个包含左上角坐标和宽高的列表 [x, y, w, h]
    # 计算目标框的四个顶点
    left = gt_rect_value[0]
    top = gt_rect_value[1]
    right = gt_rect_value[0] + gt_rect_value[2]
    bottom = gt_rect_value[1] + gt_rect_value[3]

    # 计算对角线长度
    diagonal_length = torch.sqrt(torch.tensor((gt_rect_value[2] ** 2 + gt_rect_value[3] ** 2)).float())
    # 设置固定距离，这里您已经定义为对角线长度的10%
    fixed_distance = 0.1 * diagonal_length

    # 计算对角线方向
    diagonal_direction = torch.tensor([gt_rect_value[2] / diagonal_length, gt_rect_value[3] / diagonal_length],
                                      device='cuda')

    # 在每个顶点外侧放置一个额外的点
    for i in range(M):
        # 每一组4个点后，距离加倍
        if i % 4 == 0:
            fixed_distance *= 2
        # 根据顶点的位置和对角线方向计算额外点的坐标
        x = torch.ones((1, 1), device=torch.device('cuda')) \
            * (left - diagonal_direction[0] * fixed_distance if (i % 4) % 2 == 0 else right + diagonal_direction[
            0] * fixed_distance) * x_resize
        y = torch.ones((1, 1), device=torch.device('cuda')) \
            * (bottom + diagonal_direction[1] * fixed_distance if (i % 4) > 1 else top - diagonal_direction[
            1] * fixed_distance) * y_resize
        grid_y = torch.cat([grid_y, y], dim=1)
        grid_x = torch.cat([grid_x, x], dim=1)

    xy = torch.stack([grid_x, grid_y], dim=-1)  # B, N*N, 2

    _, S, C, H, W = rgbs.shape

    print_stats('rgbs', rgbs)

    # preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    # trajs_e = preds[-1]
    # print_stats('trajs_e', trajs_e)
    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
        # print('working on keypoint %d/%d' % (n+1, N))
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:, 0] = xy[:, n]  # B, 1, 2  # set first position
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs[:, cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:, -1].unsqueeze(1).repeat(1, 8 - S_local, 1, 1, 1)], dim=1)

            preds, preds_anim, vis_e, stats = model(traj_e[:, cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, )

            vis_e = torch.sigmoid(vis_e)  # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:, cur_frame:end_frame] = xys[:, :S_local]

            found_skip = False
            thr = 0.9
            si_last = 8 - 1  # last frame we are willing to take
            si_earliest = 1  # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis_e[0, si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:, :, n] = traj_e

    pad = 50
    rgbs = F.pad(rgbs.reshape(B * S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H + pad * 2,
                                                                                            W + pad * 2)
    trajs_e = trajs_e + pad

    if sw is not None and sw.save_this:
        linewidth = 1

        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                     cmap='spring', linewidth=linewidth)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1]) * -0.5,
                                     cmap='spring', linewidth=linewidth)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

        # write to disk, in case that's more convenient
        wide_list = list(wide_cat.unbind(1))
        wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        wide_list = [Image.fromarray(wide) for wide in wide_list]

        # 使用 os.path.split() 分割路径，获取最后两级子目录
        subfold_split = subfolder.split('/')
        output_subfolder = os.path.join(subfold_split[-2], subfold_split[-1])

        # 构造输出目录的路径
        output_directory = './MyOutput/' + output_subfolder + '/' + RGBTmodel + '_traj_demo/'

        # 确保输出目录存在，如果不存在则创建它
        os.makedirs(output_directory, exist_ok=True)

        # 构造输出文件路径
        out_fn = os.path.join(output_directory, 'out_%d.gif' % sw.global_step)
        wide_list[0].save(out_fn, save_all=True, append_images=wide_list[1:])
        print('saved %s' % out_fn)

        # # alternate vis
        # sw.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs2', trajs_e[0:1], vis_e[0:1],
        #                          utils.improc.preprocess_color(rgbs[0:1]))

        # animation of inference iterations
        rgb_vis = []
        for trajs_e_ in preds_anim:
            trajs_e_ = trajs_e_ + pad
            rgb_vis.append(
                sw.summ_traj2ds_on_rgb('', trajs_e_[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                       cmap='spring', linewidth=linewidth, only_return=True))
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return trajs_e - pad


def main():
    # the idea in this file is to run the model on some demo images, and return some visualizations

    exp_name = '00'  # (exp_name is used for logging notes that correspond to different runs)

    init_dir = 'reference_model'

    ## choose hyps
    B = 1
    S = 8
    H, W = 512, 640
    N = 16 + 8  # number of po'/;'nts to track

    # 指定数据集根目录
    if RGBTmodel == 'infrared' or RGBTmodel == 'visible':
        dataset_root = 'E:/PIPs/Anti-UAV-RGBT/test'
    else:
        dataset_root = 'E:/UNIFusion-main/outputs/Anti-UAV-RGBT/test'

    # 获取所有子文件夹的列表
    subfolders = [f.path for f in os.scandir(dataset_root) if f.is_dir()]

    # 遍历每个子文件夹
    for subfolder in subfolders:
        subsubfolder = os.path.join(subfolder, RGBTmodel)
        print('folr_path', subsubfolder)

        # 拼接 'infrared.json' 文件路径

        gt_json_file = os.path.join(subfolder, f'{RGBTmodel}.json')

        filenames = glob.glob(os.path.join(subsubfolder, '*.jpg'))
        # filenames = sorted(filenames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


        print('num_img', len(filenames))
        max_iters = len(filenames) // S  # run each unique subsequence

        log_freq = 200  # when to produce visualizations

        ## autogen a name
        model_name = "%02d_%d_%d" % (B, S, N)
        model_name += "_%s" % exp_name
        import datetime
        model_date = datetime.datetime.now().strftime('%H_%M_%S')
        model_name = model_name + '_' + model_date
        print('model_name', model_name)

        log_dir = 'logs_demo'
        writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

        global_step = 0

        model = Pips(stride=4).cuda()
        parameters = list(model.parameters())
        if init_dir:
            _ = saverloader.load(init_dir, model)
        global_step = 0
        model.eval()

        # create a json for trajectory
        tra_exist = [0] * len(filenames)
        # 初始化一个存储轨迹坐标数据的列表
        all_trajs_e = []

        while global_step < max_iters:

            read_start_time = time.time()

            global_step += 1

            sw_t = utils.improc.Summ_writer(
                writer=writer_t,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq / 2),
                just_gif=True)

            try:
                rgbs = []
                exist_flag = False

                for s in range(S):
                    iter_fn = (global_step - 1) * S + s

                    if exist_flag == False:
                        # 检查 'infrared.json' 文件是否存在
                        if os.path.exists(gt_json_file):
                            with open(gt_json_file, 'r') as json_file:
                                data = json.load(json_file)

                                # 获取 "exist" 键的值，如果键不存在，则返回一个默认的空列表
                                exist_values = data.get("exist", [])

                                # 检查 "exist" 键的第 iter_fn 个元素是否为 1
                                if len(exist_values) >= iter_fn + 1 and exist_values[iter_fn] == 1:
                                    print(f"Exist value at iter_fn {iter_fn} is 1")
                                    exist_flag = True

                                    # 读取第 iter_fn 个数组的四个数字列表
                                    gt_rect_values = data.get("gt_rect", [])

                                    # 检查 "gt_rect" 数组是否包含第 iter_fn 个子数组
                                    if len(gt_rect_values) > iter_fn:
                                        gt_rect_value = gt_rect_values[iter_fn]
                                        if gt_rect_value == [0,0,0,0] or gt_rect_value == []:
                                            exist_flag = False
                                            continue
                                        # 检查子数组是否包含四个数字
                                        if len(gt_rect_value) == 4:
                                            print("gt_rect numbers:", gt_rect_value)
                                else:
                                    continue
                        else:
                            print(f"Error: '%s.json' file not found in {RGBTmodel, subfolder}")

                    # set the  tre json exist
                    tra_exist[iter_fn:global_step * S] = [1] * (S - s)

                    fn = filenames[iter_fn]
                    if s == 0:
                        print('start frame', fn)
                    im = imageio.imread(fn)

                    # 调整图像尺寸为512x640
                    im = resize(im, (H, W), anti_aliasing=True)

                    # ToDO: delete 255
                    im = (255 * im).astype('uint8')

                    im = im.astype(np.uint8)

                    rgbs.append(torch.from_numpy(im).permute(2, 0, 1))
                if exist_flag == False:
                    # 添加一组50帧的轨迹，数值为0
                    zero_trajs = torch.zeros(1, S, N, 2, device='cuda') # 1, 50, 20, 2
                    all_trajs_e.append(zero_trajs.tolist())
                    continue

                rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)  # 1, S, C, H, W

                read_time = time.time() - read_start_time
                iter_start_time = time.time()

                with torch.no_grad():
                    trajs_e = run_model(model, rgbs, N, sw_t, subfolder, gt_rect_value)

                    print('trajs_e shape', trajs_e.shape)
                    # print('trajs_e', trajs_e[0: 1])

                    # 获取第二维度的当前大小
                    current_size = trajs_e.shape[1]
                    padding_size = S - current_size

                    if padding_size > 0:
                        # 创建填充坐标的张量
                        padding_tensor = torch.zeros(1, padding_size, N, 2, device=trajs_e.device)
                        trajs_e = torch.cat((trajs_e, padding_tensor), dim=1)

                        # print('Modified trajs_e shape', trajs_e.shape)
                    # 累积每次迭代得到的S帧的轨迹坐标数据
                    all_trajs_e.append(trajs_e.tolist())

                iter_time = time.time() - iter_start_time
                print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                    model_name, global_step, max_iters, read_time, iter_time))
            except FileNotFoundError as e:
                print('error', e)

        # 创建一个字典，包含"exist"键和累积的轨迹坐标数据
        tra_data = {"exist": tra_exist, "trajs_e": all_trajs_e}

        # 将字典转换为JSON格式的字符串
        json_data = json.dumps(tra_data, indent=2)

        # 使用 os.path.split() 分割路径，获取最后两级子目录
        subfold_split = subfolder.split('/')
        output_subfolder = os.path.join(subfold_split[-2], subfold_split[-1])

        # 构造输出文件路径
        json_output_path = './MyOutput/' + output_subfolder + '/' + RGBTmodel + '_tra.json'

        # 确保目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

        # 将JSON数据写入文件
        with open(json_output_path, 'w') as file:
            file.write(json_data)

    writer_t.close()


if __name__ == '__main__':
    main()
