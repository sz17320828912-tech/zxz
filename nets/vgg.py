import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    #   用于初始化‘VGG’模型
    def __init__(self, features, num_classes=1000):  #features就是我们定义的网络结构
        super(VGG, self).__init__()
        self.features = features   #这行代码将传入的features参数赋值给类的属性self.features，用于表示网络的特征提取部分。
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))   #构建平均池化层，它将输入的特征图转换成固定大小的（7，7）
        self.classifier = nn.Sequential(   #构建分类器
            nn.Linear(512 * 7 * 7, 4096),  #这是一个全连接层，输入大小为512 * 7 * 7，输出大小为4096
            nn.ReLU(True),
            nn.Dropout(),    #Dropout层，用于防止过拟合，随机将输入张量的部分元素置零。
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()   #这是一个私有方法，用于初始化模型的权重。

    #   该方法定义了VGG模型的前向传播过程。
    def forward(self, x): #定义一个名为‘forward’的方法，他是VGG的前向传播函数
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :4 ](x)   #将输入x传递给模型的特征提取部分的前四个子模块，并将结果赋值给feat1。
        feat2 = self.features[4 :9 ](feat1)   #这行代码将feat1作为输入传递给模型的特征提取部分的第五到第九个子模块，并将结果赋值给feat2。
        feat3 = self.features[9 :16](feat2)   #将feat2作为输入传递给模型的特征提取部分的第十到第十六个子模块，并将结果赋值给feat3。
        feat4 = self.features[16:23](feat3)   #将feat3作为输入传递给模型的特征提取部分的第十七到第二十三个子模块，并将结果赋值给feat4。
        feat5 = self.features[23:-1](feat4)   #将feat4作为输入传递给模型的特征提取部分的第二十四到倒数第二个子模块（不包括最后一个子模块），并将结果赋值给feat5。
        return [feat1, feat2, feat3, feat4, feat5]

    #   这是一个私有方法，用于初始化模型的权重
    def _initialize_weights(self):   #定义一个名为‘_initialize_weights’私有方法，来初始化模型的权重，它没有接受任何参数。
        for m in self.modules():   #遍历模型中的所有模块
            if isinstance(m, nn.Conv2d):   #用于检查当前模块是否是nn.Conv2d类型的卷积层。
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #如果当前模块是卷积层，那么使用Kaiming正态分布初始化该层的权重。
                # mode='fan_out'表示权重初始化是根据输出通道数的fan-out方式，nonlinearity='relu'表示激活函数是ReLU。
                if m.bias is not None:   #如果卷积层有偏置项，则将偏置项初始化为0。
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):   #用于检查当前模块是否是nn.BatchNorm2d类型的批归一化层。
                nn.init.constant_(m.weight, 1)   #如果当前模块是将批归一化层的权重参数初始化为1。
                nn.init.constant_(m.bias, 0)   #将批归一化层的偏置参数初始化为0。
            elif isinstance(m, nn.Linear):   #用于检查当前模块是否是nn.Linear类型的全连接层。
                nn.init.normal_(m.weight, 0, 0.01)   # 如果当前模块是全连接层，将全连接层的权重参数初始化为均值为0、标准差为0.01的正态分布。
                nn.init.constant_(m.bias, 0)   #将全连接层的偏置参数初始化为0。

#用于创建VGG模型的层结构，包括卷积层和池化层
def make_layers(cfg, batch_norm=False, in_channels = 3):
    #这是一个函数定义，它接受三个参数：cfg表示层的配置列表，batch_norm表示是否使用批归一化，默认为False，in_channels表示输入通道数，默认为3。
    layers = []   #创建一个空列表，用于存储网络的层。
    for v in cfg:#对列表进行循环
        if v == 'M': #如果是M，则代表要进行最大池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]   #将一个最大池化层添加到网络的层列表中，采用2x2的池化核和步幅为2。
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            #创建一个卷积层，输入通道数为in_channels，输出通道数为v，卷积核大小为3x3，填充为1，保持输入输出大小一致。
            if batch_norm:   #如果指定使用批归一化
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]   #将卷积层、批归一化层和ReLU激活函数添加到网络的层列表中。
            else:   # 如果不使用批归一化。
                layers += [conv2d, nn.ReLU(inplace=True)]   #将卷积层和ReLU激活函数添加到网络的层列表中。
            in_channels = v   #更新输入通道数为当前输出通道数，以便下一层使用。
    return nn.Sequential(*layers)   # 返回一个包含所有层的序列模块。
# 512,512,3 -（完成两次通道为64的卷积）-> 512,512,64
# -（进行最大池化，长和宽会被压缩，通道数不会变化）-> 256,256,64 -（进行两次通道为128的卷积，且步长都=1）-> 256,256,128
# -（进行最大池化）-> 128,128,128 -（完成三次通道为256的卷积）-> 128,128,256
# -（进行最大池化）--> 64,64,256 -（完成三次通道为512的卷积）->64,64,512
# -(进行最大池化)-> 32,32,512 -（完成三次通道为512的卷积）-> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}#按照列表构建VGG16函数
#定义了一个字典cfgs，其中键为'D'，值为一个VGG16网络的配置列表。


def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    #函数调用了 VGG 类来构建一个 VGG 模型。用于创建VGG模型的层。cfgs["D"]是一个包含层配置的字典，
    # 而make_layers函数则根据这些配置创建层,**kwargs 用于传递任意数量的额外参数给 VGG 类
    if pretrained:   #检查 pretrained 参数是否为 True。
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        #如果 pretrained 为 True，则从指定的 URL 加载预训练的 VGG16，并将其存储在 state_dict 变量 模型的权重中。
        model.load_state_dict(state_dict)
    
    del model.avgpool
    del model.classifier
    #这两行代码删除了模型中的 avgpool 层和 classifier 层。avgpool 层通常是全局平均池化层，classifier 层则是模型的分类器（通常是一系列的全连接层）。
    # 这样做的目的可能是在使用预训练模型进行迁移学习时，将原始的分类器替换为适合新任务的新分类器。
    return model   #返回修改后的模型。
