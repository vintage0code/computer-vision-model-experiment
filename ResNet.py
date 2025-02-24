from functools import partial
import torch.nn as nn
from src import Scripts, Template

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(Template):
    def __str__(self):
        return "implement a ResNet for dog task, the though is attach dropout layer to deal with overfit."

    def __init__(self, params, block, num_block, task='classification'):
        super().__init__(params, task)

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.n_classes)
        self.drop_out = nn.Dropout(0.4)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        :param block: block type, basic block or bottleneck block
        :param out_channels: output depth channel number of this layer
        :param num_blocks: how many blocks per layer
        :param stride: the stride of the first block of this layer
        :return: return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = Scripts.pipeline(x, self.conv1, self.conv2_x, self.conv3_x,
                               self.conv4_x, self.conv5_x, self.avg_pool)
        feature = out.view(out.size(0), -1)
        out = Scripts.pipeline(feature, self.drop_out, self.fc)

        if self.training:
            return out
        else:
            return out, feature


# a ResNet 18 object
resnet18 = partial(ResNet, block=BasicBlock, num_block=[2, 2, 2, 2])

# a ResNet 34 object
resnet34 = partial(ResNet, block=BasicBlock, num_block=[3, 4, 6, 3])

# # a ResNet 50 object
resnet50 = partial(ResNet, block=BottleNeck, num_block=[3, 4, 6, 3])

# a ResNet 101 object
resnet101 = partial(ResNet, block=BottleNeck, num_block=[3, 4, 23, 3])

# a ResNet 152 object
resnet152 = partial(ResNet, block=BottleNeck, num_block=[3, 8, 36, 3])


if __name__ == '__main__':
    from src import argument_setting
    model = resnet18(params=argument_setting, task='classification')
    print(model.structure)


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""