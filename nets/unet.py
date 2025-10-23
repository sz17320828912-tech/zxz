import torch
import torch.nn as nn

# from torchsummary import summary

from nets.resnet import resnet50
from nets.vgg import VGG16
# from nets.attention import se_block, cbam_block,eca_block,CoordAtt,CoTAttention,AttentionModule,gnconv,S2Attention


# from resnet import resnet50
# from vgg import VGG16
# from attention import se_block, cbam_block,eca_block,CoordAtt,CoTAttention,AttentionModule,gnconv,S2Attention
# from swin_transformer import swin_tiny_patch4_window7_224


# attention_blocks = [se_block, cbam_block,eca_block]
# attention_blocks[0]
# attention_blocks[1]
# attention_blocks[2]
#上采样
#   定义了上采样类unetUp，其初始化函数中接收输入通道数in_size和输出通道数out_size
class unetUp(nn.Module):
    #   __init__ 方法初始化了一个 U-Net 中的上采样模块。
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()   #in_size 和 out_size，分别表示输入通道数和输出通道数。
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)   #这一行定义了一个卷积层 conv1，使用 nn.Conv2d 类来创建。
        # 它将输入特征图的通道数从 in_size 转换为 out_size，使用大小为3x3的卷积核，padding=1表示使用1个单位的零填充。
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)   #这一行定义了上采样层 up，使用双线性插值法进行上采样，
        # scale_factor=2 表示将输入特征图的长和宽都放大2倍。
        self.relu   = nn.ReLU(inplace = True)

    #   forward 方法定义了该模块的前向传播过程，该过程包括上采样、卷积和激活操作。
    def forward(self, inputs1, inputs2):

        #有两个输入inputs1和inputs2
        # 先把input2进行上采样，得到一个长和宽的扩张，然后和inputs1进行堆叠,堆叠之后再就像两次卷积的操作
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)   #将 inputs2 上采样两倍（使用之前定义的 up 层），然后将其与 inputs1 按通道维度拼接起来，得到一个特征图。
        outputs = self.conv1(outputs)   #这一行将拼接后的特征图输入到第一个卷积层 conv1 中进行卷积操作。
        outputs = self.relu(outputs)   #这一行将卷积后的特征图输入到ReLU激活函数中进行激活。
        outputs = self.conv2(outputs)   #这一行将ReLU激活后的特征图输入到第二个卷积层 conv2 中进行卷积操作。
        outputs = self.relu(outputs)   #这一行再次将卷积后的特征图输入到ReLU激活函数中进行激活。
        return outputs

#加强特征提取网络
class Unet(nn.Module):
    #   __init__ 方法初始化了整个 U-Net 模型
    #   包括选择主干网络（VGG 或 ResNet）、上采样模块以及最终的分类卷积层
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'resnet50'):
        #这是类的构造函数 __init__，它初始化了一个 Unet 实例。
        # 它接受三个参数：num_classes 表示输出类别数，默认为 21；
        # pretrained 表示是否使用预训练模型，默认为 False；
        # backbone 表示选择的主干网络，可以是 'vgg' 或 'resnet50'，默认为 'vgg'。

        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        #如果选择的主干网络是 VGG，就实例化一个 VGG16 模型，并将其赋值给 self.vgg。
        # 同时，根据 VGG16 模型的结构，设置输入特征图的通道数为 [192, 384, 768, 1024]。

        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        #如果选择的主干网络是 ResNet50，就实例化一个 ResNet50 模型，并将其赋值给 self.resnet。
        # 同时，根据 ResNet50 模型的结构，设置输入特征图的通道数为 [192, 512, 1024, 3072]。
        # elif backbone == "swin_tiny_patch4_window7_224":
        #     self.resnet = resnet50(pretrained = pretrained)
        #     in_filters  = [192, 512, 1024, 3072]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        #如果选择的主干网络既不是 'vgg' 也不是 'resnet50'，则抛出一个 ValueError。

        out_filters = [64, 128, 256, 512]  #out_filters对应了每次获得的unetUp的结果，得到了一个通道数是多少

        # upsampling
        #接下来是四个 unetUp 层的实例化，每个 unetUp 层都包含一个上采样操作和两个卷积层。
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])     # 输出结果获得一个 64,64,512 的特征层
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])     # 输出结果获得一个 128,128,256 的特征层
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])     # 输出结果获得一个 56,256,128 的特征层
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])     # 输出结果获得一个 512,512,64 的特征层
        #利用最后的（512，512，64）这个有效特征层进行预测

        #如果选择的主干网络是 ResNet50，则实例化一个序列模块 up_conv，
        # 该模块包含上采样操作和两个卷积层。否则，up_conv 被设置为 None。
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        # 利用1*1的卷积，对512*512*64的特征层进行一个通道数的调整，将其通道数调整为num_classes,
        # 也就相当于把我们输入的特征层的每一个特征点进行分类（也就是对我们输入的图片每一个像素点进行分类）

        self.backbone = backbone
        # #添加cbam注意力机制
        # self.up1_attention = CoordAtt(64,64)
        # self.up2_attention = CoordAtt(128,128)
        # self.up3_attention = CoordAtt(256,256)
        # self.up4_attention = CoordAtt(512,512)
        # self.up1_attention = cbam_block(64)
        # self.up2_attention = cbam_block(128)
        # self.up3_attention = gnconv(256)
        # self.up4_attention = S2Attention(512)

    #   forward 方法定义了 U-Net 模型的前向传播过程，其中包括从主干网络中提取特征并通过上采样模块连接这些特征。
    def forward(self, inputs):#根据选择的主干网络，从 VGG 或 ResNet 中提取特征。
        # inputs = inputs[:, :10, :, :]  # 选择第4、第3和第2个通道

        if self.backbone == "vgg":   #如果 backbone 是 "vgg"，则调用 VGG16 模型的前向传播函数，获取多个特征层
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":   #如果是 "resnet50"，则调用 ResNet50 模型的前向传播函数，同样获取多个特征层
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        # print('------------feat-------------')
        # print('feat1:', feat1.shape)
        # print('feat2:', feat2.shape)
        # print('feat3:', feat3.shape)
        # print('feat4:', feat4.shape)
        # print('feat5:', feat5.shape)
        #
        # print('------------up----------------')
        # print('input:',inputs.shape)

        #cbam注意力机制
        #将提取的特征通过多个 unetUp 层进行上采样和连接操作，这些层的结果被传递给下一个层，用于进一步上采样和连接。
        #特征融合操作
        up4 = self.up_concat4(feat4, feat5)
        # up4 = self.up4_attention(up4)
        # print('up4:', up4.shape)
        up3 = self.up_concat3(feat3, up4)
        # up3 = self.up3_attention(up3)
        # print('up3:', up3.shape)
        up2 = self.up_concat2(feat2, up3)
        # up2 = self.up2_attention(up2)
        # print('up2:', up2.shape)
        up1 = self.up_concat1(feat1, up2)
        # up1 = self.up1_attention(up1)
        # print('up1:', up1.shape)

        # 额外的上采样和卷积（仅在使用 resnet50 时）
        if self.up_conv != None:
            up1 = self.up_conv(up1)
            # print(up1.shape)
        final = self.final(up1)
        # print(final.shape)

        return final

    #freeze_backbone 和 unfreeze_backbone 方法分别用于冻结和解冻主干网络的参数。
    def freeze_backbone(self):   #冻结主干网络的参数
        if self.backbone == "vgg":
            #   对 VGG 模型的参数进行遍历，self.vgg.parameters() 返回模型中所有参数的迭代器
            for param in self.vgg.parameters():
                #   将参数的 requires_grad 属性设置为 False，这意味着这些参数在训练过程中不会更新梯度，即被冻结
                param.requires_grad = False

        elif self.backbone == "resnet50":
            #   对 ResNet50 模型的参数进行遍历，
            for param in self.resnet.parameters():
                #   将参数的 requires_grad 属性设置为 False，以冻结这些参数。
                param.requires_grad = False

    def unfreeze_backbone(self):   #解冻主干网络的参数
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        #它检查选择的主干网络是否为 "vgg"。
        # 如果是，它遍历 vgg 模型的所有参数，并将它们的 requires_grad 属性设置为 False，以使它们不再计算梯度。

        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        #如果选择的主干网络是 "resnet50"，则同样遍历 resnet 模型的所有参数，并将它们的 requires_grad 属性设置为 False

def main():
    # 设置一些超参数
    num_classes = 21
    input_channels = 3
    input_size = 512

    # 创建Unet实例
    model = Unet(num_classes=num_classes, pretrained=False, backbone='vgg')
    print(model)

    # 打印模型结构
    # print("Model Summary:")
    # summary(model, input_size=(input_channels, input_size, input_size))

    # 创建一个随机输入张量
    inputs = torch.randn(1, input_channels, input_size, input_size)

    # 运行前向传播过程
    with torch.no_grad():
        outputs = model(inputs)

    # 打印输出形状
    print(f"Output shape: {outputs.shape}")


if __name__ == "__main__":
    main()
