import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
from typing import Literal, Sequence, Union, List
import os
import re
import scipy.io as sio
from PIL import Image
import torchvision.transforms as transforms

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""


class DogDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, *, use_type: Literal['train', 'test', 'file'], root='.',
                 task: Literal['bounding_box', 'classification'] = 'bounding_box',
                 sampling=False, split_rate=0.65):
        super().__init__()
        self.image_path = os.path.join(root, 'ImageNetDogs/images')
        self.pos_ano_path = os.path.join(root, 'ImageNetDogs/annotations')
        self.split_rate = split_rate

        lists = sio.loadmat(f"{self.image_path[:-7]}/lists/{use_type}_list.mat")
        self.image_file_list = np.array(list(map(lambda file: file.tolist(), lists['file_list'].flatten()))).flatten()
        self.ano_file_list = [f.split('.')[0] for f in self.image_file_list]
        self.labels = lists['labels'].flatten() - 1
        if sampling:
            self.image_file_list, self.labels = self.__sub_sampling()

        self.task = task

    def __len__(self):
        return len(self.labels)

    @property
    def classes_name(self):
        unsorted_cls, idx = np.unique([f.split('-')[1].split('/')[0] for f in self.image_file_list], return_index=True)
        pac = dict(zip(unsorted_cls, idx))
        sorted_dict = sorted(pac.items(), key=lambda item: item[1])
        sorted_cls_list = list(dict(sorted_dict).keys())

        return np.array(sorted_cls_list)

    def __sub_sampling(self) -> (Sequence[str], Sequence[int]):
        """
        Stratified sampling throughout the data set for purposes such as training testing.
        self.split_rate controls the percentage of data set segmentation.
        """
        file_list, labels_list = np.array([]), np.array([])

        for i in range(len(set(self.labels))):
            category_idx = np.where(self.labels == i)[0]
            end = int(len(category_idx) * self.split_rate)
            category_idx = category_idx[:end]

            file_list = np.concatenate((file_list, self.image_file_list[category_idx]))
            labels_list = np.concatenate((labels_list, self.labels[category_idx]))

        return file_list, labels_list.astype(dtype=int)

    @staticmethod
    def _get_xy(ano_file: str) -> Sequence[int]:
        with open(ano_file) as f:
            ano_content = f.read()
        x_min = int(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', ano_content)[0])
        x_max = int(re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', ano_content)[0])
        y_min = int(re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', ano_content)[0])
        y_max = int(re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', ano_content)[0])
        return x_min, y_min, x_max, y_max

    def get_a_group_neighbour_data(self, start: int, end: int):
        image_files = map(lambda path: os.path.join(self.image_path, path), self.image_file_list[start:end])
        slice_data = [DogDataset.transform(Image.open(file).convert('RGB')) for file in image_files]
        return torch.stack(slice_data), self.labels[start:end]

    def get_dataset_box(self, idx) -> (str, Sequence[int], str):
        ano_file = os.path.join(self.pos_ano_path, self.ano_file_list[idx])
        xy = self._get_xy(ano_file)

        image_file = os.path.join(self.image_path, self.image_file_list[idx])
        cls_label = self.classes_name[self.labels[idx]]
        print("ano_file: {}\nimage_file: {}\nxy: {}".format(ano_file, image_file, xy))

        return image_file, xy, str(cls_label)

    def show_box(self, idx):
        image_file, xy, label = self.get_dataset_box(idx)
        xy = torch.tensor(xy).unsqueeze(0)
        self.draw_box(image_file, xy, [label], "yellow")

    @staticmethod
    def draw_box(image_file: str, box: Tensor, label: List[str], color: Union[str, List[str]] = 'yellow'):
        import torchvision.transforms as trans
        from torchvision.io import read_image
        from torchvision.utils import draw_bounding_boxes

        def auto_adapt_font_size(pos):
            good_font_size = {"area": 43510, "font_size": 30}
            x_min, y_min, x_max, y_max = pos[0][0], pos[0][1], pos[0][2], pos[0][3]
            area = (x_max - x_min) * (y_max - y_min)
            area = area * 4 if area < 1e4 else area

            predicted_size = int(area * (good_font_size["font_size"] / good_font_size["area"]))
            return predicted_size if predicted_size < good_font_size["font_size"] else good_font_size["font_size"]

        image = read_image(image_file)
        box = box.unsqueeze(0) if len(box.shape) == 1 else box

        font = 'visualize_file/URWBookman-Light.otf'
        font = font if os.path.exists(font) else None
        font_size = auto_adapt_font_size(box) if font else None

        image_with_boxes = draw_bounding_boxes(image, box, colors=color, width=3, labels=label,
                                               font_size=font_size, font=font)
        image_with_boxes_pil = trans.ToPILImage()(image_with_boxes)
        image_with_boxes_pil.show()
        return image_with_boxes_pil

    @staticmethod
    def box_resize(box: Sequence[Union[int, float]], ori_image_size: Sequence[int],
                   target_image_size: int = 224, resize_back: bool = False) -> Tensor:
        if resize_back:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            ori_image_w, ori_image_h = ori_image_size[0], ori_image_size[1]
            w, h = x_max - x_min, y_max - y_min
            new_x_min, new_y_min = ori_image_w * x_min / target_image_size, ori_image_h * y_min / target_image_size
            new_w, new_h = ori_image_w * w / target_image_size, ori_image_h * h / target_image_size
            new_box = torch.tensor([new_x_min, new_y_min, new_x_min + new_w, new_y_min + new_h])
        else:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            ori_image_w, ori_image_h = ori_image_size[0], ori_image_size[1]
            w, h = x_max - x_min, y_max - y_min
            new_w, new_h = w / ori_image_w * target_image_size, h / ori_image_h * target_image_size
            new_x_min, new_y_min = x_min / ori_image_w * target_image_size, y_min / ori_image_h * target_image_size
            new_box = torch.tensor([new_x_min, new_y_min, new_x_min + new_w, new_y_min + new_h])
        return new_box

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_path, self.image_file_list[idx])
        image = Image.open(image_file).convert('RGB')
        if self.task == 'bounding_box':
            ano_file = os.path.join(self.pos_ano_path, self.ano_file_list[idx])
            box = self.box_resize(torch.tensor(self._get_xy(ano_file)), image.size)
            return DogDataset.transform(image), self.labels[idx], box
        return DogDataset.transform(image), self.labels[idx]


if __name__ == '__main__':
    haha = DogDataset(use_type='test')
    a, b, c = haha[0]
    print(c)
    # haha.show_box(100)


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""
