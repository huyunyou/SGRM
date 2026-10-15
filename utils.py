from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from filter import FastGuidedFilter

def get_image(path):
    """
    Reads and returns RGB image, (1,3,H,W).
    """
    image = torch.from_numpy(np.array(Image.open(path))).float()
    image = image / torch.max(image)
    image = torch.movedim(image, -1, 0).unsqueeze(0).cuda()
    return image


def get_image_me(path, height=None, width=None, flag=False):
    if flag is True:
        image = Image.open(path).convert('RGB') #imread(path, mode='RGB')
    else:
        image = Image.open(path).convert('L')

    if height is not None and width is not None:
        #image = imresize(image, [height, width], interp='nearest')
        image = np.array(image.resize((height, width)))
        #image = image[image.shape[0]//4:, :]
    else:
        #image = cv2.resize(image,(224, 224),interpolation=cv2.INTER_NEAREST)

        #image = np.array(image.resize((224,224)))
        image = np.array(image)

        #image = image[image.shape[0]//4:, :]

    return image


def get_stokes_vector_image(path):
    '''
    input: I0,I60,I120
    output: I,Q,U,DOP
    '''
    I0 = get_image_me(path + '/0-degree.bmp').astype(np.float32)
    I60 = get_image_me(path + '/60-degree.bmp').astype(np.float32)
    I120 = get_image_me(path + '/120-degree.bmp').astype(np.float32)

    I = (I0 + I60 + I120) * (2.0 / 3.0)
    Q = (2 * I0 - I60 - I120) * 2.0 / 3.0
    U = (I60 - I120) * 2.0 / np.sqrt(3)

    eps = 1e-5
    I[I == 0] += eps

    DOP = (np.sqrt(np.power(Q, 2) + np.power(U, 2))) / I
    # AOP = 0.5*np.atan2(U,Q)

    if np.isnan(DOP).any():
        print("警告: DOP 矩阵中存在 NaN，请检查分母 I 是否包含零或负值!")
    if np.any(I == 0):
        print("存在精确的零！")


    img = DOP
    minner = np.amin(img)
    img = img - minner
    maxer = np.amax(img)
    img = img / maxer
    img = img.astype(np.float32)
    DOP_r = img
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
    DOP_d = img

    img = I
    minner = np.amin(img)
    img = img - minner
    maxer = np.amax(img)
    img = img / maxer
    img = img.astype(np.float32)
    I = img
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(I).unsqueeze(0)  # Shape: [1, C, H, W]
    I = img

    return I,Q,U,DOP_d,DOP_r

def get_single_dolp_image(path_name):
    '''
    input: dolp
    output: 0-1 的dolp
    '''
    DOP = get_image_me(path_name+'/DOP.bmp').astype(np.float32)

    img = DOP
    minner = np.amin(img)
    img = img - minner
    maxer = np.amax(img)
    img = img / maxer
    img = img.astype(np.float32)
    # transform = transforms.Compose([transforms.ToTensor()])
    # img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
    DOP_d = img

    return DOP_d

def get_single_dolp_image_20260807_simu(path_name):
    '''
    input: dolp
    output: 0-1 的dolp
    '''
    DOP = get_image_me(path_name).astype(np.float32)

    img = DOP
    minner = np.amin(img)
    img = img - minner
    maxer = np.amax(img)
    img = img / maxer
    img = img.astype(np.float32)
    DOP_r = img
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
    DOP_d = img

    return DOP_r,DOP_d

def get_v_component(img_hsv):
    """
    Assumes (1,3,H,W) HSV image.
    """
    return img_hsv[:,-1].unsqueeze(0)


def replace_v_component(img_hsv, v_new):
    """
    Replaces the V component of a HSV image (1,3,H,W).
    """
    img_hsv[:,-1] = v_new
    return img_hsv


def interpolate_image(img, H, W):
    """
    Reshapes the image based on new resolution.
    """
    return F.interpolate(img, size=(H,W))


def get_coords(H, W):
    """
    Creates a coordinates grid for INF.
    """
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float().cuda()
    return coords

# # Fourier feature mapping
# def input_mapping(x, B):
#   if B is None:
#     return x
#   else:
#     x_proj = (2.*np.pi*x) @ B.T
#     return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

def get_patches(img, KERNEL_SIZE):
    """
    Creates a tensor where the channel contains patch information.
    """
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).cuda()

    ks = KERNEL_SIZE
    sigma = 3       # 5，5，3，3，3，3
    # 创建坐标网格 (ks x ks)
    center = ks // 2  # 中心点坐标 (3,3)
    x = torch.arange(ks) - center  # 行坐标: [-3, -2, ..., 3]
    y = torch.arange(ks) - center  # 列坐标: [-3, -2, ..., 3]
    # 生成网格并计算平方距离
    X, Y = torch.meshgrid(x, y, indexing='ij')
    squared_dist = X ** 2 + Y ** 2
    # 计算高斯核
    A = torch.exp(-squared_dist / (2 * sigma))

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()),0,i,j] = A[i,j]

    pad = nn.ReflectionPad2d(KERNEL_SIZE//2)
    im_padded = pad(img)

    extracted = torch.nn.functional.conv2d(im_padded, kernel, padding=0).squeeze(0)

    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr, y_lr, x_hr, r=1):
    """
    Applies the guided filter to upscale the predicted image.
    """
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr
