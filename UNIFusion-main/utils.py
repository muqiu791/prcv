import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
from time import time

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.autograd import Variable
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import torch.nn.functional as F
import matplotlib as mpl
from torchvision import datasets, transforms


def gradient(x):
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=1);
    # showTensor(gradMap);
    return gradMap;


def gradient2(x):
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(64,64,1,1)    
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=1);
    # showTensor(gradMap);
    return gradMap;


def sumPatch(x, k):
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = np.ones((2 * k + 1, 2 * k + 1));
    kernel = kernel / (1.0 * (2 * k + 1) * (2 * k + 1));
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(1, 1, 1, 1)
    weight = nn.Parameter(data=kernel, requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=k);
    # showTensor(gradMap);
    return gradMap;


def sumPatch2(x, k):
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = np.ones((2 * k + 1, 2 * k + 1));
    kernel = kernel / (1.0 * (2 * k + 1) * (2 * k + 1));
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(64, 64, 1, 1)
    weight = nn.Parameter(data=kernel, requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x, weight=weight, stride=1, padding=k);
    # showTensor(gradMap);
    return gradMap;


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    image = np.array(image)/255;
    # print(path)
    # print("Image type:", type(image))
    # print("Image values:\n", image)
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])

        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()

    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000',
                                                                 '#8B0000'], 256)


def save_images(path, data):
    # if isinstance(paths, str):
    #     paths = [paths]
    #
    # t1 = len(paths)
    # t2 = len(datas)
    # assert (len(paths) == len(datas))

    # if prefix is None:
    #     prefix = ''
    # if suffix is None:
    #     suffix = ''

    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    imsave(path, data)

    # for i, path in enumerate(paths):
    #     data = datas[i]
    #     # print('data ==>>\n', data)
    #     if data.shape[2] == 1:
    #         data = data.reshape([data.shape[0], data.shape[1]])
    #     # print('data reshape==>>\n', data)
    #
    #     name, ext = splitext(path)
    #     name = name.split(sep)[-1]
    #
    #     path = join(save_path, prefix + suffix + ext)
    #     print('data path==>>', path)
    #
    #     # new_im = Image.fromarray(data)
    #     # new_im.show()
    #
    #     imsave(path, data)

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """

    Cr = torch.from_numpy(Cr)
    Cb = torch.from_numpy(Cb)

    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    start = time()
    temp = (im_flat + bias).mm(mat)
    end = time()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out