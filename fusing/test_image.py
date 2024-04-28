# test phase
import glob
import shutil
import imageio
import torch
import os
from torch.autograd import Variable
from net import GhostFusion_net
import utils
from utils import gradient,tensor_save_rgbimage
from scipy.misc import imread, imsave, imresize
from args_fusion import args
import numpy as np
import time
import cv2
import pywt
from skimage import img_as_ubyte

RGBTmodel = 'fuseIRVIS'  # 'fuseIRVIS' 'fuseIR_UnaffVIS'

# input order should follow the order of "irBase,viBase,irDetail,viDetail" far, near,  far, near
def _generate_fusion_mf(model, imgBase1, imgBase2, imgDetail1, imgDetail2):
    imgGradDetail1 = gradient(imgDetail1);
    imgGradDetail2 = gradient(imgDetail2);
    shape = imgBase1.shape;
    imgGradDetail1 = torch.abs(imgGradDetail1);
    imgGradDetail2 = torch.abs(imgGradDetail2);
    en_GradDetail1 = model.encoder(imgGradDetail1);
    en_GradDetail2 = model.encoder(imgGradDetail2);
    focusMap1 = model.fusion(en_GradDetail1, en_GradDetail2, strategy_type='AGL1')[0];
    focusMap2 = torch.ones(1, 1, shape[2], shape[3]) - focusMap1;
    fBase = imgBase1 * focusMap1 + imgBase2 * focusMap2;
    fDetail = imgDetail1 * focusMap1 + imgDetail2 * focusMap2;
    return fBase, fDetail


def load_model(path, input_nc, output_nc):
    nest_model = GhostFusion_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

    nest_model.eval()
    # nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):

    en_r = model.encoder(img1)
    en_v = model.encoder(img2)

    f = model.fusion(en_r, en_v, strategy_type=strategy_type)

    img_fusion = model.decoder(f);

    return img_fusion[0]



def run_demo(model, irBase_path, irDetail_path, visBase_path, visDetail_path, output_path_root, index, BS, DS, mode):

    irBase_img = utils.get_test_images(irBase_path, height=None, width=None, mode='RGB')
    irDetail_img = utils.get_test_images(irDetail_path, height=None, width=None, mode='L')
    visBase_img = utils.get_test_images(visBase_path, height=None, width=None, mode='RGB')
    visDetail_img = utils.get_test_images(visDetail_path, height=None, width=None, mode='L')

    # Rgb2YCrCb
    irBase_img,Cb_irBase,Cr_irBase = utils.RGB2YCrCb(irBase_img)
    visBase_img,Cb_visBase,Cr_visBase = utils.RGB2YCrCb(visBase_img)

    # Cr,Cb fusion
    def wavelet_fusion(image1, image2):
        # 对图像执行小波变换
        coeffs1 = pywt.dwt2(image1, 'bior1.3')
        coeffs2 = pywt.dwt2(image2, 'bior1.3')

        # 融合近似系数
        fused_approx = (coeffs1[0] + coeffs2[0]) / 2

        # 使用最大规则融合详细系数
        fused_detail = np.maximum(coeffs1[1], coeffs2[1])

        # 重构融合后的图像
        fused_image = pywt.idwt2((fused_approx, fused_detail), 'bior1.3')

        return fused_image

    # 假设你有Cr_irBase，Cb_irBase，Cr_visBase，Cb_visBase作为输入通道
    fused_Cr = wavelet_fusion(Cr_irBase, Cr_visBase)
    fused_Cb = wavelet_fusion(Cb_irBase, Cb_visBase)


    # dim = img_ir.shape
    if args.cuda:
        irBase_img = irBase_img.cuda(args.device)
        irDetail_img = irDetail_img.cuda(args.device)
        visBase_img = visBase_img.cuda(args.device)
        visDetail_img = visDetail_img.cuda(args.device)
        model = model.cuda(args.device);
    irBase_img = Variable(irBase_img, requires_grad=False)
    irDetail_img = Variable(irDetail_img, requires_grad=False)
    visBase_img = Variable(visBase_img, requires_grad=False)
    visDetail_img = Variable(visDetail_img, requires_grad=False)
    # multifocus
    if (BS == 'AGL1' and DS == 'AGL1'):
        fusedBase, fusedDetail = _generate_fusion_mf(model, irBase_img, visBase_img, irDetail_img, visDetail_img);
    else:
        # strategy_type_list = strategy_type_list = ['AVG', 'L1','SC','MAX','AGL1']
        # Base L1
        fusedBase = _generate_fusion_image(model, BS, irBase_img, visBase_img)
        fusedDetail = _generate_fusion_image(model, DS, irDetail_img, visDetail_img)

    fusedBase = fusedBase[0].cpu();
    fusedBase = fusedBase.squeeze().squeeze();
    fusedBase = fusedBase.numpy();
    fusedBase = fusedBase * 255;

    # Detail max

    fusedDetail = fusedDetail[0].cpu();
    fusedDetail = fusedDetail.squeeze().squeeze();
    fusedDetail = fusedDetail.numpy();
    fusedDetail = fusedDetail * 255;

    # sub
    fusedDetail = fusedDetail - np.mean(fusedBase);

    # finalFuseResult
    Y_fusedFinalResult = fusedDetail + fusedBase;
    Y_fusedFinalResult = torch.from_numpy(Y_fusedFinalResult).unsqueeze(0).unsqueeze(0)
    formatted_index = "{:04d}".format(index)
    fusedFinalResult_output_path = os.path.join(output_path_root, RGBTmodel,
                                                f"{RGBTmodel}{formatted_index}.jpg")
    # 检查输出路径是否存在，如果不存在则创建
    if not os.path.exists(os.path.dirname(fusedFinalResult_output_path)):
        os.makedirs(os.path.dirname(fusedFinalResult_output_path))

    RGBResult=utils.YCbCr2RGB(Y_fusedFinalResult/255.0,fused_Cb,fused_Cr)

    RGBResult = RGBResult.squeeze()
    fusedFinalResult = RGBResult.numpy()*255


    imsave(fusedFinalResult_output_path, fusedFinalResult.transpose(1, 2, 0).astype('uint8'));
    print(fusedFinalResult_output_path)


def main():
    current_dir = os.path.dirname(__file__)
    dataset_root = os.path.abspath(os.path.join(current_dir,'../Anti-UAV-RGBT/test'))
    test_root_path = os.path.abspath(os.path.join(current_dir,'../tracking/Myoutput/Anti-UAV-RGBT/test'))

    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1', 'SC', 'MAX',
                          'AGL1']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    BS = strategy_type_list[1];
    DS = strategy_type_list[3];


    in_c = 1
    if in_c == 1:
        out_c = in_c
        mode = 'L'
        model_path = args.model_path_gray
    else:
        out_c = in_c
        mode = 'RGB'
        model_path = args.model_path_rgb
    print('model_path:', model_path)
    print('mode:', mode)

    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        for subdir in os.listdir(test_root_path):
            print('sub',subdir)
            subdir_path = os.path.join(test_root_path, subdir)

            if os.path.isdir(subdir_path):
                ir_path = os.path.join(subdir_path, 'IR')
                if RGBTmodel == 'fuseIRVIS':
                    vis_path = os.path.join(subdir_path, 'aff_VIS')
                else:
                    vis_path = os.path.join(subdir_path, 'VIS')

            filenames = glob.glob(os.path.join(ir_path, '*.jpg'))
            print(len(filenames))
            max_iters = len(filenames) // 2

            # 使用 os.path.split() 分割路径，获取最后两级子目录
            subfold_split = subdir_path.split('/')
            output_subfolder = os.path.join(subfold_split[-2], subfold_split[-1])

            output_path = './outputs/' + output_subfolder + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for i in range(max_iters):
                irBase_path = os.path.join(ir_path, f'IRBase{str(i).zfill(4)}.jpg')
                irDetail_path = os.path.join(ir_path, f'IRDetail{str(i).zfill(4)}.jpg')
                visBase_path = os.path.join(vis_path, f'VISBase{str(i).zfill(4)}.jpg')
                visDetail_path = os.path.join(vis_path, f'VISDetail{str(i).zfill(4)}.jpg')

                run_demo(model,irBase_path,irDetail_path, visBase_path, visDetail_path, output_path, i, BS, DS, mode)

            # 找到对应的gt_json_path文件
            gt_json_path = dataset_root + '/' + subdir + '/infrared.json'
            # 构建目标路径
            target_json_path = os.path.join(output_path, f'{RGBTmodel}.json')

            # 复制文件并重命名
            shutil.copyfile(gt_json_path, target_json_path)
            print(f'Copied {gt_json_path} to {target_json_path}')

    print('Done......')


if __name__ == '__main__':
    main()