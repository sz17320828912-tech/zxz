import random

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        # image = image.convert('RGB')
        return image


# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#


def resize_image(image, size):
    # print("resize_image", image)
    # print("resize_image", image.shape)
    ih, iw, channels = image.shape
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    # image = image.resize((nw, nh), Image.BICUBIC)
    # new_image = Image.new('RGB', size, (128, 128, 128))
    # new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    # 使用numpy进行图像缩放
    resized_image = np.resize(image, (nh, nw, channels))
    # 创建输出图像并填充灰色
    new_image = np.full((h, w, channels), 128, dtype=np.uint8)
    # 计算粘贴位置
    paste_x = (w - nw) // 2
    paste_y = (h - nh) // 2
    # 使用numpy进行图像粘贴
    new_image[paste_y:paste_y+nh, paste_x:paste_x+nw, :] = resized_image

    return new_image, nw, nh

# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        'mobilevit_s': 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt',
        'mobilevit_ef_one': 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt',
        'mobilevit_ef': 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
