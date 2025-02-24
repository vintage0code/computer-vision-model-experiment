from Dataset import DogDataset
from VggNet import VGG16
from ResNet import resnet18
from GoogLeNet import GoogLeNetForDogTask
from ObjectDetection import GoogLeNetForDogDetection
from src import fix, Scripts, deal_with_folder
import argparse
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch import nn

title = """
********************************************************************************************
***********************实验对比和模型调优：基于改良网络的狗品种分类任务*************************
"""

# TODO: 得出了两个主要结论：第一就是无需数据增强；第二就是原生划分比列不太利于模型快速收敛
# TODO： 由于模型参数是在每个batch上更新一次，那么batch_size太小，会使得模型过于关注该批次数据，就容易陷入过拟合
# TODO： 动量的计算公式中包含了当前梯度信息和上一批次的梯度信息，而上一批次的梯度信息又包含了上上批次的梯度信息...故总体来看，过往梯度信息
#        是随距离指数衰减的影响到当前总梯度。比如整个训练需要更新1000次，设动量权重为0.9，那么第一次的动量值对第1000次时梯度值的影响只有0.9的1000次方


def train(the_model, *args, **kwargs):
    """
    model training process based on pytorch_lightning framework.
    :param the_model: the model to be trained. no instantiation required
    :param args: the parameters transfer to the model
    :param kwargs: the parameters transfer to the model
    :return: nothing return
    """
    assert not isinstance(the_model, nn.Module),\
        "parameters 'the_model' do not need to be instantiated and must belong to the class nn.Module."

    # Parse parameters that are not passed to the model
    parsed_args, data_set = kwargs.pop('parsed_args', None), kwargs.pop('data_set', None)

    torch.manual_seed(2019)
    rate = 0.6  # split train
    num_tran, num_rest = fix(data_size := len(data_set), int(data_size * rate), int(data_size * (1 - rate)))
    the_train_data, the_rest_data = random_split(data_set, [num_tran, num_rest])
    # train/test/valid ratio: 0.6/0.2/0.2
    rate = 0.5  # split val-test
    num_val, num_test = fix(data_size := len(the_train_data), int(data_size * rate), int(data_size * (1 - rate)))
    the_val_data, the_test_data = random_split(the_train_data, [num_val, num_test])

    if parsed_args.accelerator == 'gpu':
        train_loader = DataLoader(the_train_data, batch_size=parsed_args.batch, num_workers=8, persistent_workers=True)
        valid_loader = DataLoader(the_val_data, batch_size=parsed_args.batch, num_workers=8, persistent_workers=True)
        test_loader = DataLoader(the_test_data, batch_size=parsed_args.batch, num_workers=8, persistent_workers=True)
    else:
        train_loader = DataLoader(the_train_data, batch_size=parsed_args.batch)
        valid_loader = DataLoader(the_val_data, batch_size=parsed_args.batch)
        test_loader = DataLoader(the_test_data, batch_size=parsed_args.batch)

    logger = WandbLogger(project=parsed_args.project_name,
                         name=parsed_args.run_name, save_dir=deal_with_folder('check_points'))
    logger.log_hyperparams(argument_setting)

    model = the_model(*args, **kwargs)
    print("the total parameters of the model is: ", Scripts.count_parameters(model))
    print(model.structure)
    trainer = pl.Trainer(max_epochs=parsed_args.epochs, accelerator=parsed_args.accelerator, logger=logger)
    # trainer = pl.Trainer(max_epochs=parsed_args.epochs, accelerator=parsed_args.accelerator,
    #                      callbacks=[EarlyStopping(monitor='val_loss', patience=3)], logger=logger)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.test(model=model, dataloaders=test_loader)
    if parsed_args.save_model:
        try:
            model_name = the_model.func.__name__
        except AttributeError:
            model_name = the_model.__name__
        torch.save(model, f"./{deal_with_folder('saved_model')}/{model_name}_model.pth")
    wandb.finish()
    torch.cuda.empty_cache()


# run command: python main.py -p vgg10 -rn exp1 -m vgg
if __name__ == '__main__':
    model_list = ['google_net', 'vgg', 'res_net', 'google_net2detection']
    parser = argparse.ArgumentParser(description='The setting of dog task')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--classes', '-c', type=int, default=120, help='number of classes')
    parser.add_argument('--project_name', '-p', required=True, help='wandb\'s project name')
    parser.add_argument('--run_name', '-rn', type=str, required=True, help='run name in a project')
    parser.add_argument('--save_model', '-sm', action='store_false', help='save model parameters')
    parser.add_argument('--epochs', '-e', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--accelerator', '-a', default='gpu', help='choose accelerator for trainer')
    parser.add_argument('--task', '-t', type=str, default='bounding_box',
                        choices=['classification', 'bounding_box'], help='task to you preference')
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=model_list, help='model to you preference')
    parser.add_argument('--specific_loss_fn', '-slf', action='store_true',
                        help='whether needs replace the loss function')
    parser.add_argument('--no_scheduler', '-ns', action='store_true',
                        help='whether learning without scheduler')

    the_args = parser.parse_args()

    argument_setting = {
        "hyp_params": {
            "lr": the_args.lr, "no_scheduler": the_args.no_scheduler
        },
        "config": {
            "n_classes": the_args.classes, "device": the_args.accelerator,
            "specific_loss_fn": the_args.specific_loss_fn
        }
    }

    Scripts.plot_parameters_bar(models := [GoogLeNetForDogTask, VGG16, resnet18, GoogLeNetForDogDetection])

    data_all = DogDataset(use_type='file')
    train(the_model=models[model_list.index(the_args.model)], params=argument_setting,
          task=the_args.task, parsed_args=the_args, data_set=data_all)


author = """
**************************************作者:2100100717王耀斌************************************
**********************************************************************************************
"""
