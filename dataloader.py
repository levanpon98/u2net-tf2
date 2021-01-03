import os
import glob
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from skimage import io, transform, color

from models.u2net import u2net


class Dataset(tf.keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 batch_size=16,
                 rescale=320,
                 random_crop=(288, 288),
                 to_tensor_flag=0):

        self.data_path = data_path
        self.batch_size = batch_size
        self.rescale_size = rescale
        self.random_crop_size = random_crop
        self.load_raw_data()
        self.to_tensor_flag = to_tensor_flag

    def load_raw_data(self):
        meta_data = os.path.join(
            self.data_path, 'CelebA-HQ-to-CelebA-mapping.txt')
        relation = pd.read_csv(meta_data, delim_whitespace=True,
                               na_filter=False)
        relation = shuffle(relation)
        self.indexes = list(item['idx'] for idx, item in relation.iterrows())

    def load_image_mask(self, data_path, data):
        imgs = []
        masks = []
        for idx, row in data.iterrows():
            img_name = row['idx']
            img = os.path.join(data_path, 'CelebA-HQ-img/' +
                               str(img_name) + '.jpg')
            mask = os.path.join(
                data_path, 'CelebAMaskHQ-mask/' + str(img_name) + '.png')

            imgs.append(img)
            masks.append(mask)

        return imgs, masks

    def rescale(self, image, label):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.rescale_size * h / w, self.rescale_size
        else:
            new_h, new_w = self.rescale_size, self.rescale_size * w / h
        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (self.rescale_size, self.rescale_size), mode='constant')
        label = transform.resize(label, (self.rescale_size, self.rescale_size), mode='constant', order=0,
                                 preserve_range=True)

        return image, label

    def random_crop(self, image, label):

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.random_crop_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return image, label

    def normalize(self, image, label):
        tmpLbl = np.zeros(label.shape)
        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        # change the color space
        if self.to_tensor_flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                    np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                    np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                    np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                    np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                    np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                    np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.to_tensor_flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                    np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                    np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                    np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        # return
        return tmpImg, tmpLbl

    def data_generation(self, idx):
        image = io.imread(os.path.join(
            self.data_path, 'CelebA-HQ-img', str(idx) + '.jpg'))
        label = io.imread(os.path.join(
            self.data_path, 'CelebAMaskHQ-mask', str(idx) + '.png'))

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        image, label = self.rescale(image, label)  # rescale
        image, label = self.random_crop(image, label)  # random crop
        image, label = self.normalize(image, label)

        return image, label

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        labels = []
        for idx in indexes:
            image, label = self.data_generation(idx)
            images.append(image)
            labels.append(label)

        # images, labels = np.array(images), np.array(labels)
        images = np.transpose(images, (0, 2, 3, 1))
        labels = np.transpose(labels, (0, 2, 3, 1))

        # images = tf.convert_to_tensor(images, np.float32)
        # labels = tf.convert_to_tensor(labels, np.float32)
        return images, labels


if __name__ == '__main__':
    dataset = Dataset('/home/ponlv/work/data/CelebAMask-HQ')
    x, y = dataset[0]
    out = u2net(x, 32)
    print(out.shape)
    print(x.shape)
    print(y.shape)
    # # print(next(dataset))
    # data_path = '/home/ponlv/work/data/CelebAMask-HQ'
    # image = io.imread(os.path.join(
    #     data_path, 'CelebA-HQ-img', str('0.jpg')))
    # label = io.imread(os.path.join(
    #     data_path, 'CelebAMaskHQ-mask', str('0.jpg').replace('jpg', 'png')))
    #
    # if 3 == len(image.shape) and 2 == len(label.shape):
    #     label = label[:, :, np.newaxis]
    # elif 2 == len(image.shape) and 2 == len(label.shape):
    #     image = image[:, :, np.newaxis]
    #     label = label[:, :, np.newaxis]
    #
    # image, label = dataset.rescale(image, label)
    # image, label = dataset.random_crop(image, label)
    # image, label = dataset.normalize(image, label)
    #
    # print(label)
    # print(label.shape)
