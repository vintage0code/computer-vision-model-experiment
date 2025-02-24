from src import Scripts, argument_setting, Template
import torch.nn as nn

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


class VGG16(Template):
    def __str__(self):
        return ("implement a VGG10 for dog task, the though is reduce conv block "
                " and grow the output feature to adapt 120 classes. The large full connect layer also be replaced.")

    def __init__(self, params, task='classification'):
        super().__init__(params=params, task=task)
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = self.vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = self.vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = self.vgg_conv_block([128, 256], [256, 256], [3, 3], [1, 1], 2, 2)
        self.layer4 = self.vgg_conv_block([256, 512], [512, 512], [3, 3], [1, 1], 2, 2)
        self.layer5 = self.vgg_conv_block([512], [1024], [3], [1], 2, 2)

        # Final layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.layer8 = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        out = Scripts.pipeline(x, self.layer1, self.layer2, self.layer3, self.layer4)
        vgg16_feature_maps = self.layer5(out)   # features if you want to use somewhere else
        out = Scripts.pipeline(vgg16_feature_maps, self.avg_pool)
        feature = out.view(out.size(0), -1)
        out = Scripts.pipeline(feature, self.dropout, self.layer8)

        if self.training:
            return out
        else:
            return out, feature

    @staticmethod
    def conv_layer(channel_in, channel_out, k_size, p_size):
        layer = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=k_size, padding=p_size),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        )
        return layer

    @staticmethod
    def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
        layers = [VGG16.conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
        layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
        return nn.Sequential(*layers)

    @staticmethod
    def vgg_fc_layer(size_in, size_out):
        layer = nn.Sequential(
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.ReLU()
        )
        return layer


if __name__ == '__main__':
    vgg16 = VGG16(argument_setting, "classification")
    print(vgg16.structure)


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""