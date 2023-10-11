TTM
现已完善框架

当前用在了视频分类任务中：

调用之前请在config/base.json 中的dataset字典定义相关路径和参数
训练batch(由本地显存决定)，epoch请自行修改,tensorbaord名字，请在train字典中修改

usage: python train.py
断点权重保存在了./checkpoint
tensorboard保存在了./log

把tensorboard2excel也融入进去了
现在只要一改完config的参数，直接运行main.py，会自动训练、验证、生成excel文件表，
并且每次训练的最小损失、收敛batch和测试集测试准确率都会保存在experiment.txt文件夹下。

目前还在更新......