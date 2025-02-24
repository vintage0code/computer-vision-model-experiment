import torch.nn.functional as f
import torch
import torch.nn as nn
from src import Template, Scripts

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return Scripts.pipeline(x, self.conv, self.bn, self.relu)


class Front(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x):
        # (N,3,224,224) -> (N,64,112,112) -> (N,64,56,56) -> (N,64,56,56) -> (N,192,56,56) -> (N,192,28,28)
        return Scripts.pipeline(x, self.conv1, self.max_pool1, self.conv2, self.conv3, self.max_pool2)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_1_1, ch3x3_1, ch3x3_2_1, ch3x3_2, pool_ch, /):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_1_1, kernel_size=1),
            BasicConv2d(ch3x3_1_1, ch3x3_1, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_2_1, kernel_size=1),
            BasicConv2d(ch3x3_2_1, ch3x3_2, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_ch, kernel_size=1)
        )

    def forward(self, x):
        # 输入(N,Cin,Hin,Win)
        branch1 = self.branch1(x)  # (N,C1,Hin,Win)
        branch2 = self.branch2(x)  # (N,C2,Hin,Win)
        branch3 = self.branch3(x)  # (N,C3,Hin,Win)
        branch4 = self.branch4(x)  # (N,C4,Hin,Win)
        outputs = [branch1, branch2, branch3, branch4]
        # (N,C1+C2+C3+C4,Hin,Win) -> (N,ch1x1 + ch3x3_1 + ch3x3_2 + pool_ch,Hin,Win)
        return torch.cat(outputs, 1)


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1:(N,512,14,14) -> (N,512,4,4) -> (N,128,4,4); aux2: (N,528,14,14) -> (N,528,4,4) -> (N,128,4,4)
        x = Scripts.pipeline(x, self.averagePool, self.conv)
        x = torch.flatten(x, 1)  # (N,2048)
        x = f.dropout(x, 0.5, training=self.training)
        x = f.relu(self.fc1(x))  # (N,1024)
        x = f.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)  # (N,num_classes)
        return x


class GoogLeNet(Template):
    def __init__(self, params, task='classification', aux_logit=False):
        super().__init__(params, task)
        self.aux_logit = aux_logit

        self.front = Front()
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logit:
            self.aux1 = InceptionAux(512, self.n_classes)
            self.aux2 = InceptionAux(528, self.n_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        aux1, aux2 = None, None
        # (N,3,224,224) -> (N,192,28,28) -> (N,256,28,28) -> (N,480,28,28) -> (N,480,14,14) -> (N,512,14,14)
        x = Scripts.pipeline(x, self.front, self.inception3a, self.inception3b, self.max_pool3, self.inception4a)
        if self.training and self.aux_logit:
            aux1 = self.aux1(x)

        # (N,512,14,14) -> (N,512,14,14) -> (N,528,14,14)
        x = Scripts.pipeline(x, self.inception4b, self.inception4c, self.inception4d)
        if self.training and self.aux_logit:
            aux2 = self.aux2(x)

        # # (N,832,14,14) -> (N,832,7,7) -> (N,832,7,7) -> (N,1024,7,7) -> (N,1024,1,1)
        x = Scripts.pipeline(x, self.inception4e, self.max_pool4, self.inception5a, self.inception5b, self.avg_pool)
        x = torch.flatten(x, 1)  # (N,1024)
        x = Scripts.pipeline(x, self.dropout, self.fc)   # (N,num_classes)
        if self.training and self.aux_logit:
            return x, aux2, aux1
        else:
            return x


class GoogLeNetForDogTask(Template):
    def __str__(self):
        return "implement a GoogLeNet for dog task, the though is reduce inception module to adapt 120 classes"

    def __init__(self, params, task='classification', aux_logit=False):
        super().__init__(params, task)
        self.aux_logit = aux_logit

        self.front = Front()
        # (N,ch1x1 + ch3x3_1 + ch3x3_2 + pool_ch,Hin,Win)
        # TODO：主要分为细节学习和完形学习，细节学习后接着完形学习，分为三个层次
        self.inception3a = Inception(192, 64, 128, 240, 32, 144, 64)
        self.inception3b = Inception(512, 128, 224, 360, 64, 216, 64)
        self.max_pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(768, 256, 512, 400, 432, 240, 128)
        self.inception4b = Inception(1024, 384, 618, 960, 468, 576, 128)
        self.max_pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        if self.aux_logit:
            self.aux = InceptionAux(2048, self.n_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # TODO: 特征的个数也实验一下，因为用了drop_put，试一试1024和2048（因为resnet上经验告诉2048配合dropout的效果不错）
        self.fc = nn.Linear(2048, self.n_classes)

    def forward(self, x):
        aux = None
        # (N,3,224,224) -> (N,192,28,28) -> (N,512,28,28) -> (N,768,28,28) -> (N,768,14,14)
        x = Scripts.pipeline(x, self.front, self.inception3a, self.inception3b, self.max_pool3)
        if self.training and self.aux_logit:
            aux = self.aux(x)

        # (N,768,14,14) -> (N,1024,14,14) -> (N,2048,14,14) -> (N,2048,7,7) -> (N,2048,1,1)
        x = Scripts.pipeline(x, self.inception4a, self.inception4b, self.max_pool4, self.avg_pool)
        features = torch.flatten(x, 1)  # (N,2048)
        x = Scripts.pipeline(features, self.dropout, self.fc)   # (N,num_classes)

        if self.training and self.aux_logit:
            return x, aux
        elif self.training:
            return x
        else:
            return x, features


if __name__ == '__main__':
    from src import argument_setting
    net = GoogLeNetForDogTask(argument_setting)
    print(net.structure)

author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""