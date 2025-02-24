import warnings
import torch
from torch import nn
from GoogLeNet import GoogLeNetForDogTask
from src import argument_setting, Scripts


# TODO: 有几点可以改进的地方，一是box的回归可以用w和h的形式，这样会让模型更清楚坐标对任务的意义。
#  第二是试试将box归一化。第三是head试试弄几个卷积层做深一点
# TODO：目前的结论是归一化后的loss尺度太小了，可能不利于学习
class GoogLeNetForDogDetection(GoogLeNetForDogTask):
    def __str__(self):
        return ("implement a GoogLeNet for dog detection task."
                "The though is make use of pretrain model as the backbone"
                "and the detection model could go on a regression after that.")

    def __init__(self, params, task='bounding_box',
                 model_path='saved_model/best/GoogLeNetForDogTask_model.pth'):
        super().__init__(params=params, task=task)
        try:
            self.load_state_dict(torch.load(model_path).state_dict())
            for param in self.parameters():
                param.requires_grad = False
        except FileNotFoundError:
            warnings.warn("not found the weight file of pretrain model, "
                          "or get a wrong file path, so we continue to train a new one.")

        self.fc = None
        self.fc_for_classification = nn.Linear(2048, self.n_classes)

        self.pos_encoder = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc_for_detection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # (N,3,224,224) -> (N,192,28,28) -> (N,512,28,28) -> (N,768,28,28) -> (N,768,14,14)
        x = Scripts.pipeline(x, self.front, self.inception3a, self.inception3b, self.max_pool3)

        # (N,768,14,14) -> (N,1024,14,14) -> (N,2048,14,14)
        x = Scripts.pipeline(x, self.inception4a, self.inception4b)
        # (N,2048,14,14) -> (N,2048,7,7) -> (N,2048,1,1)
        features = torch.flatten(Scripts.pipeline(x, self.max_pool4, self.avg_pool), 1)  # (N,2048)
        classification_out = Scripts.pipeline(features, self.dropout, self.fc_for_classification)  # (N,num_classes)
        # (N,2048,14,14) -> (N,2048,7,7) -> (N,1024,7,7) -> (N,1024,1,1) -> (N,1024)
        detection_feat = torch.flatten(Scripts.pipeline(x, self.max_pool4, self.pos_encoder, self.avg_pool), 1)
        detection_out = self.fc_for_detection(detection_feat)    # (N,4)
        return detection_out, classification_out


if __name__ == '__main__':
    da = GoogLeNetForDogDetection(argument_setting)
    data = torch.randn(4, 3, 224, 224)
    d, _ = da(data)
    print(d.shape)
