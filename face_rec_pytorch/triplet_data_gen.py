import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset
import os
import numpy as np
from random import shuffle,choice,sample
import cv2
import torch
from PIL import Image
import matplotlib.pylab as plt
from skimage import exposure


def categorical_func(y, num_classes=None, dtype='float32'):
    #将输入y向量转换为数组
    y = np.array(y, dtype='int')
    #获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    #y变为1维数组
    y = y.ravel()
    #如果用户没有输入分类个数，则自行计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    #生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    #np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    #进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class TripletDataGen(Dataset):
    def __init__(self,
                 path2dataset,
                 batch_size,
                 num_ids_per_batch,
                 is_shuffle,
                 for_test,
                 *arg_):
        assert batch_size%num_ids_per_batch == 0
        self.batch_size = batch_size
        self.list_ids_names = os.listdir(path2dataset)
        self.list_ids_names.sort()
        self.list_ids_index = list(range(0,len(self.list_ids_names)))
        self.path2dataset = path2dataset
        self.is_shuffle = is_shuffle
        self.num_ids_per_batch = num_ids_per_batch
        self.num_ids = len(self.list_ids_index)
        self.for_test = for_test
        self.input_shape = (128,128)
        if self.is_shuffle:
            shuffle(self.list_ids_index)
        self.ids_index = 0

        super(TripletDataGen).__init__(*arg_)
    def __len__(self):
        return self.num_ids

    def on_epoch_end(self):
        self.ids_index = 0
        if self.is_shuffle:
            shuffle(self.list_ids_index)

    def __getitem__(self, item):
        if item>(self.num_ids//self.num_ids_per_batch):
            self.on_epoch_end()
        item = item%(self.num_ids//self.num_ids_per_batch)
        num_image_per_ids = self.batch_size//self.num_ids_per_batch

        ids_index_start = item*self.num_ids_per_batch
        ids_index_end = (item+1)*self.num_ids_per_batch
        list_images = []
        list_ids_labels =[]
        for i in range(ids_index_start,ids_index_end):
            ids_index = self.list_ids_index[i]
            path2images = os.path.join(self.path2dataset,self.list_ids_names[ids_index])
            list_image_names = os.listdir(path2images)
            num_image_current_id = len(list_image_names)

            list_image_indics = list(range(0, num_image_current_id))
            if len(list_image_names)<num_image_per_ids:
                continue
            for j in sample(list_image_indics,num_image_per_ids):
                # image = cv2.imread(os.path.join(path2images,list_image_names[j]))
                image = Image.open(os.path.join(path2images,list_image_names[j]))
                image = np.array(image)
                image = cv2.resize(image,self.input_shape)
                # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                # image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                if np.random.uniform(0,1)>0.6 and (not self.for_test):
                    m = cv2.getRotationMatrix2D(
                        (self.input_shape[1] // 2, self.input_shape[0] // 2),
                        np.random.randint(-5, 5),
                        np.random.uniform(0.9, 1.1))
                    image = cv2.warpAffine(image, m, self.input_shape)
                    image = exposure.adjust_gamma(
                        image,
                        np.random.uniform(0.9, 1.1))
                    if np.random.uniform(0,1.)>0.5:
                        image = cv2.flip(image,1)
                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                list_images.append(image)
                list_ids_labels.append(self.list_ids_index[i])

        list_images = np.array(list_images)
        list_ids_labels = np.array(list_ids_labels)

        list_ids_labels = categorical_func(
            list_ids_labels,
            num_classes=self.num_ids)

        list_images = list_images.astype(np.float32)
        # list_images -= 70.5
        # list_images /= 50.2
        list_images -= 127.5
        list_images /= 127.5

        # to torch.Tensor
        list_images = torch.from_numpy(list_images)
        list_ids_labels = torch.from_numpy(list_ids_labels)

        return list_images,list_ids_labels
                # print(image.shape)

class TripletDataGenPair(Dataset):
    def __init__(self,
                 path2dataset,
                 batch_size,
                 is_shuffle,
                 for_test,
                 **arg_):
        self.batch_size = batch_size
        self.list_ids_names = os.listdir(path2dataset)
        self.list_ids_names.sort()
        self.list_ids_index = list(range(0,len(self.list_ids_names)))
        self.path2dataset = path2dataset
        self.is_shuffle = is_shuffle
        self.num_ids = len(self.list_ids_index)
        self.for_test = for_test
        self.input_shape = (128,128)
        if self.is_shuffle:
            shuffle(self.list_ids_index)
        self.ids_index = 0

        super(TripletDataGen).__init__(**arg_)
    def __len__(self):
        return self.num_ids

    def on_epoch_end(self):
        self.ids_index = 0
        if self.is_shuffle:
            shuffle(self.list_ids_index)

    def image_prepro(self,image):
        # image = cv2.resize(image,self.input_shape)
        image = image.astype(np.float32)
        image = image / 127.5
        image = image - 1.0
        if len(image.shape)==2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.expand_dims(image,0)
        return image

    def rand_transform(self,image):
        if np.random.uniform(0, 1) > 0.7 and (not self.for_test):
            m = cv2.getRotationMatrix2D(
                (self.input_shape[1] // 2, self.input_shape[0] // 2),
                np.random.randint(-5, 5),
                np.random.uniform(0.9, 1.1))
            image = cv2.warpAffine(image, m, self.input_shape)
            image = exposure.adjust_gamma(
                image,
                np.random.uniform(0.9, 1.1))
            if np.random.uniform(0, 1.) > 0.5:
                image = cv2.flip(image, 1)

        return image

    def __getitem__(self, item):
        list_triplet_pairs = []
        for i in range(self.batch_size):
            if self.ids_index>self.num_ids-1:
                self.on_epoch_end()
            anchor_ids_index = self.list_ids_index[self.ids_index]
            negative_ids_index = (anchor_ids_index+np.random.randint(1,self.num_ids-1))%self.num_ids
            path2anchor_images = os.path.join(self.path2dataset,self.list_ids_names[anchor_ids_index])
            path2negative_images = os.path.join(self.path2dataset,self.list_ids_names[negative_ids_index])

            list_positive_images = os.listdir(path2anchor_images)
            list_negative_images = os.listdir(path2negative_images)
            if self.for_test:
                anchor_image_name,positive_image_name = \
                    sample(list_positive_images,2)
                negative_image_name = \
                    choice(list_negative_images)
            else:
                anchor_image_name, positive_image_name = sample(list_positive_images, 2)
                negative_image_name = choice(list_negative_images)
            anchor_image = cv2.imread(os.path.join(path2anchor_images,anchor_image_name))
            positive_image = cv2.imread(os.path.join(path2anchor_images,positive_image_name))
            negative_image=  cv2.imread(os.path.join(path2negative_images,negative_image_name))

            anchor_image = cv2.resize(anchor_image,self.input_shape)
            positive_image = cv2.resize(positive_image,self.input_shape)
            negative_image = cv2.resize(negative_image,self.input_shape)

            anchor_image = self.rand_transform(anchor_image)
            positive_image = self.rand_transform(positive_image)
            negative_image = self.rand_transform(negative_image)

            anchor_image = self.image_prepro(anchor_image)
            positive_image = self.image_prepro(positive_image)
            negative_image = self.image_prepro(negative_image)

            triplet_pair = np.concatenate((anchor_image,positive_image,negative_image),0)

            list_triplet_pairs.append(triplet_pair)
            self.ids_index+=1

        triplet_pairs = torch.from_numpy(np.array(list_triplet_pairs))

        return triplet_pairs  


if __name__ == '__main__':

    train_gen = TripletDataGen(
        path2dataset=r'..\CASIA-WebFace-Croped',
        batch_size=256, num_ids_per_batch=256, is_shuffle=True,for_test=False)

    for _ in range(1000):
        x, y = train_gen.__getitem__(_)
        x = x.detach().numpy()
        print(x.shape)
        for i in range(x.shape[0]):
            xr = ((x[i,0] + 1.0) * 127.5).astype(np.uint8)
            xg = ((x[i,1] + 1.0) * 127.5).astype(np.uint8)
            xb = ((x[i,2] + 1.0) * 127.5).astype(np.uint8)
            plt.subplot(1, 3, 1)
            plt.imshow(((x[i,0] + 1.0) * 127.5).astype(np.uint8))
            plt.subplot(1, 3, 2)
            plt.imshow(((x[i,1] + 1.0) * 127.5).astype(np.uint8))
            plt.subplot(1, 3, 3)
            plt.imshow(((x[i,2] + 1.0) * 127.5).astype(np.uint8))
            plt.show()
            break