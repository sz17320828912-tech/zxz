import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image


# from nets.msnet import MSNet as msnet

from nets.unet import Unet as unet

# from nets.hed_unet import HEDUNet as hedunet
# from nets.layers import Convx2

# from nets.transunet import TransUnet as transunet

# from nets.CMFPNet import CMFPNet

from utils.utils import cvtColor, preprocess_input, resize_image, show_config


# ____________________________________________________
# ____________________________________________________
# from nets.CMFPNet import CMFPNet
# from nets.unet import UNet
# from nets.HRNet_4branch import HighResolutionNet
# from nets.deeplabv3 import DeepLabV3Plus
# from nets.BHENet import BHENet
# from nets.CMFFNet import CMFFNet
# from nets.PVNet import PVNet
# from nets.transunet import TransUnet
# from nets.SegFormer import Segformer
# from nets.UnetForMer import UNetFormer

# from nets.vitcross_seg_modeling import CONFIGS
# from nets.vitcross_seg_modeling import VisionTransformer
# ____________________________________________________
# ____________________________________________________

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
#--------------------------------------------#
class Unet(object):
    #    _defaults是一个类属性，包含了模型的默认参数设置。
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        # "model_path"    : 'model_data/unet_vgg_voc.pth',
        # "model_path": 'logs_fish_cbam/ep060-loss0.112-val_loss0.120.pth',
        "model_path": 'logs/best_epoch_weights.pth',

        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        #--------------------------------#
        "backbone"      : "resnet50",
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : [256, 256],
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 1,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"          : True,

        # hedunet
        # "input_channels" : 7,
        # "base_channels" : 32,
        # "conv_block" : Convx2,
        # "padding_mode" : 'replicate',
        # "batch_norm" : True,
        # "squeeze_excitation" : True,
        # "merging" : 'attention',
        # "stack_height" : 5,
        # "deep_supervision" : True

        # transunet
        "img_dim" : 256,
        "in_channels" : 10,
        "out_channels" : 128,
        "head_num" : 4,
        "mlp_dim" : 512,
        "block_num" : 8,
        "patch_dim" : 16,
        "class_num" : 2,

        "config_name" : 'R50-ViT-B_16',

    }

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    #   __init__方法是类的初始化方法，用于初始化模型的参数和配置。
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0),(255, 255, 255),(255, 255, 240), (186, 85, 211),(0, 255, 255), (30, 144, 255), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    #   generate方法用于生成模型，并加载预训练的模型权重。
    def generate(self, onnx=False):
        #   创建了一个UNet模型的实例，并将其赋值给self.net属性。
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)
        # self.net = msnet(num_classes = self.num_classes)

        # self.net = CMFPNet(num_classes = self.num_classes)

        #
        # self.net = hedunet(input_channels=self.input_channels,
        #                    num_classes=self.num_classes,
        #                    base_channels=self.base_channels,
        #                    conv_block=self.conv_block,
        #                    padding_mode=self.padding_mode,
        #                    batch_norm=self.batch_norm,
        #                    squeeze_excitation=self.squeeze_excitation,
        #                    merging=self.merging,
        #                    stack_height=self.stack_height,
        #                    deep_supervision=self.deep_supervision)



        # ___________________________________________________
        # ___________________________________________________
        # self.net = CMFPNet(num_classes=self.num_classes)
        # self.net = UNet(n_channels=10, n_classes=2, bilinear=True)
        # self.net = HighResolutionNet(num_classes=2)
        # self.net =DeepLabV3Plus(num_classes=2)
        # self.net = BHENet()
        # self.net = PVNet()
        # self.net = CMFFNet(num_classes=2)
        # self.net = TransUnet(img_dim=self.img_dim,
        #                      in_channels=self.in_channels,
        #                      out_channels=self.out_channels,
        #                      head_num=self.head_num,
        #                      mlp_dim=self.mlp_dim,
        #                      block_num=self.block_num,
        #                      patch_dim=self.patch_dim,
        #                      class_num=self.class_num, )

        # self.net =Segformer(
        #     dims=(32, 64, 160, 256),  # 各个阶段的维度
        #     heads=(1, 2, 5, 8),  # 各个阶段的头数
        #     ff_expansion=(8, 8, 4, 4),  # 各个阶段的前馈扩展因子
        #     reduction_ratio=(8, 4, 2, 1),  # 各个阶段自注意力的降采样率
        #     num_layers=2,  # 每个阶段的层数
        #     channels=10,  # 输入图像的通道数为10
        #     decoder_dim=256,  # 解码器的维度
        #     num_classes=2  # 分割类别数
        # )

        # self.net =UNetFormer(decode_channels=64, dropout=0.1, backbone_name='swsl_resnet18',
        #                pretrained=False, window_size=8, num_classes=2)

        # config_name = 'R50-ViT-B_16'
        # config = CONFIGS[config_name]
        # self.net = VisionTransformer(config, img_size=256, num_classes=2, zero_head=True, vis=True)

        # ___________________________________________________
        # ___________________________________________________

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #   这一行加载了预训练模型的权重。torch.load()函数从磁盘加载模型权重文件
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                #   这一行将模型包装成DataParallel，这样可以在多个GPU上并行地运行模型
                self.net = nn.DataParallel(self.net)
                #   这一行将模型移动到GPU上进行计算
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    #   detect_image方法用于对输入图像进行语义分割预测。
    #   这是一个类方法的定义，它接受三个参数：
    #   image表示输入的图像，count表示是否计数每个类别的像素数量，name_classes表示类别名称。
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   输入是一张图片
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        #image       = cvtColor(image)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        #   获取输入图像的高度和宽度
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   用于对输入图像进行缩放，使其尺寸符合模型的输入尺寸要求
        #   并返回缩放后的图像数据、新的宽度和高度。
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   对图像进行预处理，包括转置和扩展维度操作，
        #   添加上batch_size维度
        # 进行transpose，pytorch要把通道转换为第一维度，就是batch_size后面的第一维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)


        with torch.no_grad():
            #   将图像数据转换成PyTorch张量。
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   将图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   这一行对预测结果进行softmax归一化，并将其从PyTorch张量转换为NumPy数组
            #   permute--将通道转成最后一维，softmax--取出每一个像素点所对应的最大概率的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize：将预测结果按照原始图像的尺寸进行插值，以恢复到原始大小
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            #   创建了一个全零数组，用于存储每个类别的像素数量。
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            #   遍历每个类别
            for i in range(self.num_classes):
                #   统计了类别i的像素数量
                num     = np.sum(pr == i)
                #   计算了类别i的像素数量占总像素数量的比例
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        #   根据mix_type的值选择不同的混合方式
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            #   这一行根据预测结果和颜色列表生成彩色分割图像。
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image
    #   get_FPS方法用于计算模型在输入图像上的预测速度（帧率）（FPS）
    #   它接受两个参数：image表示输入的图像，
    #   test_interval表示测试间隔，即在多少次循环中进行性能测试。
    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        #   记录了性能测试的开始时间
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        #   记录了性能测试的结束时间
        t2 = time.time()
        #   计算了每次预测的平均时间
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #   convert_to_onnx方法用于将PyTorch模型转换为ONNX格式，并可选择是否简化模型。
    def convert_to_onnx(self, simplify, model_path):
        import onnx
        #   调用generate方法，传入参数onnx=True，以生成ONNX模型。
        self.generate(onnx=True)
        #   创建一个形状为(1, 3, *self.input_shape)的零张量im，用于指定模型的输入。
        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        #   定义输入和输出图像的名称
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        #   使用torch.onnx.export函数将PyTorch模型导出为ONNX格式。参数说明如下
        torch.onnx.export(self.net,
                        im,   #   模型的输入数据
                        f               = model_path,   #导出的ONNX模型文件的路径
                        verbose         = False,   #控制是否输出详细信息
                        opset_version   = 12,   #指定ONNX的opset版本
                        training        = torch.onnx.TrainingMode.EVAL,   #设置模型导出时的训练模式
                        do_constant_folding = True,   #控制是否执行常量折叠优化
                        input_names     = input_layer_names,   #指定模型输入的名称
                        output_names    = output_layer_names,   #指定模型输出的名称
                        dynamic_axes    = None)   #控制是否使用动态轴

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model--使用ONNX库的load函数加载导出的ONNX模型
        onnx.checker.check_model(model_onnx)  # check onnx model--使用ONNX库的checker.check_model函数对导出的ONNX模型进行检查，确保模型的有效性

        # Simplify onnx--简化模型
        if simplify:
            import onnxsim   #导入ONNX简化工具onnx-simplifier
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'   #对简化后的模型进行断言检查，确保简化成功
            onnx.save(model_onnx, model_path)   #将简化后的ONNX模型保存到指定路径

        print('Onnx model save as {}'.format(model_path))

    #   get_miou_png方法用于计算预测结果的 mIOU（Mean Intersection over Union）。
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

#代码中还定义了一个Unet_ONNX类，用于加载 ONNX 格式的 Unet 模型并进行推理
class Unet_ONNX(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   onnx_path指向model_data文件夹下的onnx权值文件
        #-------------------------------------------------------------------#
        "onnx_path"    : 'model_data/models.onnx',
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : 21,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        #--------------------------------#
        "backbone"      : "resnet50",
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : [512, 512],
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 0,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        import onnxruntime
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)
        # 获得所有的输入node
        self.input_name     = self.get_input_name()
        # 获得所有的输出node
        self.output_name    = self.get_output_name()

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**self._defaults)

    def get_input_name(self):
        # 获得所有的输入node
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        # 获得所有的输出node
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_feed(self,image_tensor):
        # 利用input_name获得输入的tensor
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor
        return input_feed
    
    #---------------------------------------------------#
    #   对输入图像进行resize
    #---------------------------------------------------#
    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size

        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        return new_image, nw, nh

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上bawtch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        input_feed  = self.get_input_feed(image_data)
        pr          = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)[0][0]

        def softmax(x, axis):
            x -= np.max(x, axis=axis, keepdims=True)
            f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
            return f_x
        print(np.shape(pr))
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = softmax(np.transpose(pr, (1, 2, 0)), -1)
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image
