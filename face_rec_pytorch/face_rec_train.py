# -*- coding=utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import cv2
import tqdm
import sys
from mobilenetv2_torch import MobileNetV2
from triplet_data_gen import TripletDataGen
from random import shuffle, choice, randrange,sample

def decay_learning_rate(epoch):
    """
    调整学习率
    为了防止学习率过大，在收敛到全局最优点的时候会来回摆荡，所以要让学习率随着训练轮数不断按指数级下降，收敛梯度下降的学习步长。
    :param epoch: 当前epoch
    :return: 当前学习率
    """
    #2e-3
    lr = 2e-3
    return lr * (0.95 ** epoch)



def test(model, device):
    '''
    原tensorflow版本中的eval-lfw
    :param model: pytorch模型
    :param device: 当前设备(cpu or gpu)
    '''
    # 测试集路径 ..\lfw-deepfunneled-Croped
    path2dataset = r'../lfw-deepfunneled-Croped'
    cls_names = os.listdir(path2dataset)
    input_shape = (128, 128)
    from PIL import Image

    # 挑选包含多张人脸的类别
    list_multi_image_cls = []
    list_single_image_cls = []

    for cls_name in cls_names:
        path2cls = os.path.join(path2dataset, cls_name)
        list_image_names = os.listdir(path2cls)
        if len(list_image_names) > 1:
            list_multi_image_cls.append(cls_name)
        else:
            list_single_image_cls.append(cls_name)
    # 测试N对人脸，计算一对人脸之间的余弦距离，大于阈值认为不是同一个人，小于阈值认为是同一个人
    # 根据真实值，计算正确分类的人脸对数，作为准确率
    list_acc = []
    list_pos_acc = []
    list_neg_acc = []
    list_pos_dist = []
    list_neg_dist = []
    for i in range(1000):
        paris_label = 0
        if np.random.uniform(0., 1.) > 0.5:
            # 随机选择当前人脸对来自同一个人
            cls_name = choice(list_multi_image_cls)
            path2cls = os.path.join(path2dataset, cls_name)
            list_image_names = os.listdir(path2cls)
            image_name0, image_name1 = sample(list_image_names, 2)
            path2image0 = os.path.join(path2cls, image_name0)
            path2image1 = os.path.join(path2cls, image_name1)
            paris_label = 1

        else:
            # 随机选择当前人脸对来自不同人
            cls_name0, cls_name1 = sample(cls_names, 2)
            path2cls0 = os.path.join(path2dataset, cls_name0)
            path2cls1 = os.path.join(path2dataset, cls_name1)

            list_image_names0 = os.listdir(path2cls0)
            list_image_names1 = os.listdir(path2cls1)

            image_name0 = choice(list_image_names0)
            image_name1 = choice(list_image_names1)
            path2image0 = os.path.join(path2cls0, image_name0)
            path2image1 = os.path.join(path2cls1, image_name1)
            paris_label = 0

        image0 = Image.open(path2image0)
        image1 = Image.open(path2image1)

        image0 = np.array(image0)
        image1 = np.array(image1)

        # 图像预处理
        image0 = cv2.resize(image0, input_shape)
        image1 = cv2.resize(image1, input_shape)

        image0 = image0.astype(np.float32)
        image1 = image1.astype(np.float32)

        # 像素值归一化
        # image0 -= 70.5
        # image0 /= 50.2
#
        # image1 -= 70.5
        # image1 /= 50.2
        image0 -= 127.5
        image0 /= 127.5

        image1 -= 127.5
        image1 /= 127.5

        image0 = np.swapaxes(image0, 0, 2)
        image0 = np.swapaxes(image0, 1, 2)
        image1 = np.swapaxes(image1, 0, 2)
        image1 = np.swapaxes(image1, 1, 2)

        image0 = np.expand_dims(image0, 0)
        image1 = np.expand_dims(image1, 0)

        image0 = torch.from_numpy(image0).to(device)
        image1 = torch.from_numpy(image1).to(device)
        # 网络推理，获得人脸特征向量
        vec_0 = model(image0)
        vec_1 = model(image1)

        vec_0 = vec_0.to('cpu').detach().numpy()
        vec_1 = vec_1.to('cpu').detach().numpy()
        # 向量做L2归一化
        vec_0 = vec_0 / np.linalg.norm(vec_0, 2)
        vec_1 = vec_1 / np.linalg.norm(vec_1, 2)

        # 计算余弦距离
        dist = 1 - np.dot(vec_0, vec_1.T) / (np.linalg.norm(vec_0) * np.linalg.norm(vec_1))
        threshold = 0.8
        # 距离阈值，是一个先验值，可以先在要测试的数据上做简单的实验，试出一个最佳值
        if paris_label == 0:
            # paris_label ==0 表示两个人脸真实值是来自不同人
            if dist < threshold:
                # 距离小于阈值，表示网络预测这两张人脸距离是同一个人

                list_neg_acc.append(0)
                list_acc.append(0)
            else:
                list_neg_acc.append(1)
                list_acc.append(1)

            list_neg_dist.append(dist)
        else:
            if dist > threshold:
                list_pos_acc.append(0)
                list_acc.append(0)

            else:
                list_pos_acc.append(1)
                list_acc.append(1)

            list_pos_dist.append(dist)

    print('acc %.4f' % np.mean(list_acc))
    print('pos acc %.4f' % np.mean(list_pos_acc))
    print('neg acc %.4f' % np.mean(list_neg_acc))

    print('pos mean dist %.4f' % np.mean(list_pos_dist))
    print('neg mean dist %.4f' % np.mean(list_neg_dist))

    print('pos std dist %.4f' % np.std(list_pos_dist))
    print('neg std dist %.4f' % np.std(list_neg_dist))


def train():
    '''
    训练模型
    '''
    # set params
    model_weight_path = 'mobilenet_v2-b0353104.pth'
    # 预训练的mobilnetv2权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 自动判断当前环境中使用gpu or cpu
    #path2dataset = r'..\CASIA-WebFace-Croped'
    path2dataset = r'..\minidata\train'
    # 训练数据路径
    cls_names = os.listdir(path2dataset)
    num_cls = len(cls_names)
    input_shape = (128,128)

    # checkpoint
    chckpoint_path = r'./checkpoints'
    if not os.path.exists(chckpoint_path):
        os.mkdir(chckpoint_path)

    # build neural network model
    model = MobileNetV2(3, num_cls, alpha=1)
    # 初始化模型并加载权重
    pre_weight = torch.load(model_weight_path)
    pre_dict = {k: v for k, v in pre_weight.items() if 'classifier' not in k}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
    for param in model.features.parameters():
        param.requires_grad = False
    # 将模型加载到设备上
    model.to(device)

    # ..\minidata\train
    # ..\minidata\test
    # data set
    # 训练集和验证集
    gen_train = TripletDataGen(
        r'..\minidata\train',
        batch_size=256,num_ids_per_batch=16,is_shuffle=True,for_test=False)
    gen_test = TripletDataGen(
        r'..\minidata\test',
        batch_size=256, num_ids_per_batch=16, is_shuffle=True, for_test=True)

    # training setup
    # 随机种子
    torch.manual_seed(2)
    # 优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3, weight_decay=0.01)
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits
    # 总的训练轮数
    epochs = 40

    # training loop
    train_steps = len(gen_train)
    for epoch in range(1, epochs+1):
        # 设置为训练模式
        model.train()
        loss_sum = 0.0
        train_bar = tqdm.tqdm(gen_train, file=sys.stdout)
        # training step loop
        for step, data in enumerate(train_bar):
            step += 1
            # 获取训练数据并放入设备
            features, labels = data
            features, labels = features.to(device), labels.to(device)

            # inference
            predictions = model(features)
            loss = loss_func(predictions, labels)

            # backpropogation and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summarise training information
            loss_sum += loss.item()

            # set custom learning rate
            optimizer.param_groups[0]["lr"] = decay_learning_rate(epoch)

            # 训练进度可视化
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch, epochs, loss)
            if step >= train_steps:
                break
            # save model
            # save_file = 'face_rec_%s_%s.pkl' % (str(epoch), str(step))
            # torch.save(model, os.path.join(chckpoint_path, save_file), pickle_module=pickle, pickle_protocol=2)

        # evaluate model
        # 设为验证模式
        model.eval()
        with torch.no_grad():
            loss_test = 0
            acc = 0
            total = 0
            test_bar = tqdm.tqdm(gen_test, file=sys.stdout)
            # 使用与训练集相同的数据进行测试
            for step_test, data in enumerate(test_bar):
                step_test += 1
                # 获取测试数据
                features, labels = data
                features, labels = features.to(device), labels.to(device)
                # 推理模型，获取预测结果
                predictions = model(features)
                # 计算损失值loss
                loss_test += loss_func(predictions, labels).item()
                # 将预测的onehot编码转换为类别
                pred_logits = torch.max(predictions, dim=1)[1]
                labels_logits = torch.max(labels, dim=1)[1]
                # 统计预测正确的数量
                acc += torch.eq(pred_logits, labels_logits).sum().item()
                total += features.shape[0]
                # 可视化
                test_bar.desc = "test epoch[{}/{}]".format(epoch, epochs)
                if step_test >= len(test_bar):
                    break
            print('test_loss:', loss_test / step_test)
            print('test_accuracy:', acc / total)
            # 使用lfw数据进行测试
            test(model, device)
    torch.save(model,'face_rec_train.pth')
    torch.save(model.state_dict(), 'face_rec_train_dict.pth')


if __name__ == '__main__':
    train()