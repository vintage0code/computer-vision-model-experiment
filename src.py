from torch import nn
import torch
from torchmetrics import Accuracy, F1Score, Precision
from torchvision.ops import distance_box_iou_loss
import torch.optim as optim
import pytorch_lightning as pl
from abc import abstractmethod
from torch import Tensor
from Dataset import DogDataset
from typing import Union, Dict, List, Any, Sequence
import os

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


class Template(pl.LightningModule):
    def __str__(self):
        return "A template for all these networks prepare to train so that we can focus on constructing a network."

    @property
    def structure(self):
        assert self.__class__.__name__ != 'Template', "The abstractive template has no model structure."
        return self.__repr__()

    def __init__(self, params: Dict[str, Dict[str, Any]], task: str):
        super().__init__()
        self.config = params["config"]
        self.hyp_params = params["hyp_params"]

        self.use_specific_loss_fn = self.config["specific_loss_fn"]
        self.n_classes = self.config["n_classes"]
        self.lr = self.hyp_params["lr"]
        self.no_scheduler = self.hyp_params["no_scheduler"]

        self.task = task

        self.accuracy = Accuracy("multiclass", num_classes=self.n_classes)
        self.precision = Precision(task="multiclass", num_classes=self.n_classes)
        self.f1 = F1Score("multiclass", num_classes=self.n_classes, average='macro')
        self.loss_fn = self.specific_loss_fn() if self.use_specific_loss_fn else nn.CrossEntropyLoss()

        if self.task == "bounding_box":
            self.pos_loss_fn = distance_box_iou_loss
            # self.pos_loss_fn = nn.SmoothL1Loss(reduction="mean")

        self.scheduler = None

    @abstractmethod
    def forward(self, x):
        pass

    def __white_board_log(self, metrics: Dict[str, Union[float, int]], use_type: str):
        self.log(use_type + "loss", metrics["loss"], prog_bar=True, logger=True)
        del metrics["loss"]

        for key in metrics.keys():
            self.log(use_type + key, metrics[key])

    def training_step(self, batch, batch_idx):
        if self.scheduler is not None:
            self.log("learning_rate", self.scheduler.get_last_lr()[0])

        if self.task == "classification":
            x, labels = batch
            prob = self.forward(x)

            if self.__class__.__name__ == 'GoogLeNetForDogTask' and len(prob) == 2:
                main, aux = prob
                aux_weight, prob = 0.3, main

                loss = aux_weight * self.loss_fn(aux, labels) + self.loss_fn(main, labels)
            else:
                loss = self.loss_fn(prob, labels)

            acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "pcn": pcn}
        else:
            x, labels, box_labels = batch
            box_score, prob = self.forward(x)

            clsify_loss = self.loss_fn(prob, labels)
            box_loss = self.pos_loss_fn(box_score, box_labels)
            iou = box_iou(box_score, box_labels)
            loss = clsify_loss + (box_loss := box_loss.mean())

            # acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            acc, f1 = self.accuracy(prob, labels), self.f1(prob, labels)
            # metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss}
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss,
                            "sum_iou": iou.sum(), "avg_iou": iou.mean()}

        self.__white_board_log(metrics_dict, "train_")
        return loss

    def validation_step(self, batch, batch_idx):
        if self.task == "classification":
            x, labels = batch
            prob = self.forward(x)

            loss = self.loss_fn(prob, labels)

            acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "pcn": pcn}
        else:
            x, labels, box_labels = batch
            box_score, prob = self.forward(x)

            clsify_loss = self.loss_fn(prob, labels)
            box_loss = self.pos_loss_fn(box_score, box_labels)
            iou = box_iou(box_score, box_labels)
            loss = clsify_loss + (box_loss := box_loss.mean())

            # acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            acc, f1 = self.accuracy(prob, labels), self.f1(prob, labels)
            # metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss}
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss,
                            "sum_iou": iou.sum(), "avg_iou": iou.mean()}

        self.__white_board_log(metrics_dict, "val_")
        return loss

    def test_step(self, batch, batch_idx):
        if self.task == "classification":
            x, labels = batch
            prob = self.forward(x)

            loss = self.loss_fn(prob, labels)

            acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "pcn": pcn}
        else:
            x, labels, box_labels = batch
            box_score, prob = self.forward(x)

            clsify_loss = self.loss_fn(prob, labels)
            box_loss = self.pos_loss_fn(box_score, box_labels)
            iou = box_iou(box_score, box_labels)
            loss = clsify_loss + (box_loss := box_loss.mean())

            # acc, f1, pcn = self.accuracy(prob, labels), self.f1(prob, labels), self.precision(prob, labels)
            acc, f1 = self.accuracy(prob, labels), self.f1(prob, labels)
            metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss,
                            "sum_iou": iou.sum(), "avg_iou": iou.mean()}
            # metrics_dict = {"loss": loss, "acc": acc, "f1": f1, "iou_loss": box_loss}

        self.__white_board_log(metrics_dict, "test_")
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        if self.no_scheduler:
            return optimizer
        else:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # 学习率0.85倍衰减
            self.scheduler = scheduler
            return [optimizer], [scheduler]

    def specific_loss_fn(self):
        pass


class Scripts:
    @staticmethod
    def pipeline(initial_value: Union[int, float, complex, Tensor], *funcs)\
            -> Union[int, float, complex, Tensor]:
        """
        A pipeline for an input who needs to be continuous processing.
        :param initial_value: initial input
        :param funcs: finite number of processing functions
        :return: processed result
        """
        result = initial_value
        for func in funcs:
            result = func(result)
        return result

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        Count the total parameters of a model under the pytorch framework.
        :param model: a model under the pytorch framework
        :return: the total parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def plot_parameters_bar(models: List[nn.Module], save_fig: bool = False) -> None:
        import matplotlib.pyplot as plt

        data_pac = dict()
        for m in models:
            try:
                model_name = m.func.__name__
            except AttributeError:
                model_name = m.__name__

            data_pac[model_name] = Scripts.count_parameters(m(argument_setting))

        plt.bar(list(data_pac.keys()), list(values := data_pac.values()), color='sky'+'blue')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.title('differences of model parameter')
        plt.ylabel('Parameters')
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        if save_fig:
            plt.savefig('differences of model parameter.png', dpi=300)
        plt.show()


def box_iou(boxes1: Sequence[Sequence[Union[int, float]]],
            boxes2: Sequence[Sequence[Union[int, float]]], normalized_box=False):
    """
    The IOU validation metrics of every two boxes.
    :param normalized_box: if box are normalized
    :param boxes1: a batch of boxes
    :param boxes2: another batch of boxes
    :return: IOU of this batch
    """
    def single_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    if normalized_box:
        boxes1 = [DogDataset.convert((224, 224), box, only_norm=True, denormalize=True) for box in boxes1]
        boxes2 = [DogDataset.convert((224, 224), box, only_norm=True, denormalize=True) for box in boxes2]
    return torch.tensor([single_iou(box1, box2) for box1, box2 in zip(boxes1, boxes2)])


def fix(total, a: int, b: int) -> (int, int):
    """
    This function is used to replace digits lost due to type conversion to shaping.
    :param total: the a+b result before type conversion
    :param a: an int digit after type conversion
    :param b: another int digit after type conversion
    :return: the new a and b reaches the goal of a+b equals to total
    """
    if sum([a, b]) == total:
        return a, b
    elif sum([a, b]) < total:
        a += 1
        fix(total, a, b)
        return a, b
    else:
        a -= 1
        fix(total, a, b)
        return a, b


def deal_with_folder(desired_folder_path: str) -> str:
    """
    the desired folder path transformed to the available folder path.
    :param desired_folder_path: the desired folder path
    :return: the available folder path to the desired folder path
    """
    if not os.path.exists(desired_folder_path):
        os.makedirs(desired_folder_path)
    return desired_folder_path


def instance_predict(model, data_set, image_file: str, detection: bool = False,
                     det_ture_predict_compare: bool = True,
                     true_box: Sequence[Union[int, float]] = None,
                     true_cls: str = None, save_fig: bool = False) -> None:
    """
    Visualize instance prediction result for both classification and detection task.
    :param data_set: data set
    :param model: the best performance model for prediction
    :param image_file: the instance image to predict
    :param save_fig: if save the visualized figure
    :param detection: if it is detection task to visualize
    :param det_ture_predict_compare: if load dataset image to predict and compare when detection task
    :param true_box: the true box if call det_ture_predict_compare function
    :param true_cls: the true category if call det_ture_predict_compare function
    :return: nothing return
    """
    from PIL import Image, ImageDraw, ImageFont
    import torch

    # image to tensor
    img = Image.open(image_file)
    img_tensor = data_set.transform(img).unsqueeze(0)
    if detection:
        # prediction
        box, clsify = model(img_tensor)
        clsify_predict_res = torch.argmax(clsify, dim=1)
        predict_cls = data_set.classes_name[clsify_predict_res]
        # print(size := img.size)
        # box = data_set.convert(size, box.squeeze(0), denormalize=True)
        box = data_set.box_resize(box.squeeze(0), img.size, resize_back=True)
        # draw prediction true annotation result
        if det_ture_predict_compare:
            assert det_ture_predict_compare and true_box and true_cls,\
                "if wants to compare, ture box needs to be prepare."

            color, cls = ["yellow", "green"], ["P_" + predict_cls, "T_" + true_cls]
            box = torch.stack((box, torch.tensor(true_box)))
            img = data_set.draw_box(image_file, box, cls, color)
        else:
            # draw prediction result
            img = data_set.draw_box(image_file, box.int(), [predict_cls], "yellow")
    else:
        # prediction
        score, _ = model(img_tensor)
        predict_res = torch.argmax(score, dim=1)
        predict_cls = data_set.classes_name[predict_res]
        # draw prediction result
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./visualize_file/URWBookman-Light.otf', 40)
        draw.text((10, 10), text=str(predict_cls), font=font, fill=(255, 255, 0, 255))
        img.show()

    if save_fig:
        img.save(f"predicted_{predict_cls}.png")


argument_setting = {
    "hyp_params": {
        "lr": 0.01, "no_scheduler": False
    },
    "config": {
        "n_classes": 120, "simple_fine_tune": True,
        "specific_loss_fn": False
    }
}


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""