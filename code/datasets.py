from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None,
             flip = False):
    #img = Image.open(img_path).convert('RGB')
    img = Image.open(img_path).convert('L')
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    dim = 2**cfg.TREE.BRANCH_NUM * 32
    img = img.resize((dim,dim), Image.BILINEAR)
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    #img = img.point(lambda i: i / 255.)
    #print("size", img.size)
    #print("min, max", img.getextrema())
    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))
        #print(i, "size", re_img.size)
        #print(i, "min, max", re_img.getextrema())

    return ret


class XrayUncondDataset(data.Dataset):
#class ImageFolder(data.Dataset):
    def __init__(self, root, split_dir='train', custom_classes=None,
                 base_size=64, transform=None, target_transform=None):
        root = os.path.join(root, split_dir)
        classes, class_to_idx = self.find_classes(root, custom_classes)
        imgs = self.make_dataset(classes, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print('num_classes', self.num_classes)

    def find_classes(self, dir, custom_classes):
        classes = []

        for d in os.listdir(dir):
            if os.path.isdir:
                if custom_classes is None or d in custom_classes:
                    classes.append(os.path.join(dir, d))
        print('Valid classes: ', len(classes), classes)

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, classes, class_to_idx):
        images = []
        for d in classes:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[d])
                        images.append(item)
        print('The number of images: ', len(images))
        return images

    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs_list = get_imgs(path, self.imsize,
                             transform=self.transform,
                             normalize=self.norm)

        return imgs_list

    def __len__(self):
        return len(self.imgs)


class LSUNClass(data.Dataset):
    def __init__(self, db_path, base_size=64,
                 transform=None, target_transform=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            print('length: ', self.length)
        cache_file = db_path + '/cache'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
            print('Load:', cache_file, 'keys: ', len(self.keys))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.transform = transform
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        imgs = get_imgs(buf, self.imsize,
                        transform=self.transform,
                        normalize=self.norm)
        return imgs

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % \
            (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs, embedding, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

class XrayBinaryCondDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 base_size=64, condition=None, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.condition = condition

        self.dict_idx_by_class = {}
        self.class_names = None
        if self.condition == "SIDE":
            self.class_names = ["left", "right"]
        elif self.condition == "FRACTURE":
            self.class_names = ["0", "1"]
        self.dict_idx_by_class = {0:[], 1:[]}

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        #self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        #self.captions = self.load_all_captions()

        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            #class_id = np.arange(total_num)
            class_id= total_num*[-1]
            for idx, filename in enumerate(self.filenames):
                if "_{}_".format(self.class_names[0]) in filename:
                    class_id[idx] = 0
                    self.dict_idx_by_class[0].append(idx)
                elif "_{}_".format(self.class_names[1]) in filename:
                    class_id[idx] = 1
                    self.dict_idx_by_class[1].append(idx)
                #print(idx, filename, "_{}_".format(self.class_names[0]), class_id[idx])
        return class_id

    #def load_filenames(self, data_dir):
    #    filepath = os.path.join(data_dir, 'filenames.pickle')
    #    with open(filepath, 'rb') as f:
    #        filenames = pickle.load(f)
    #    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    #    return filenames

    def load_filenames(self, data_dir):
        filenames = []
        for root, _, fnames in sorted(os.walk(data_dir)):
            for fname in fnames:
                if is_image_file(fname):
                    filenames.append(fname)
        print('Load filenames : %d' % (len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]

        if cfg.COND_TYPE == 'SIDE':
            label = 0
            bbox = None
            data_dir = self.data_dir
        else:
            label = self.class_id[index]
            if self.bbox is not None:
                bbox = self.bbox[key]
                data_dir = '%s/CUB_200_2011' % self.data_dir
            else:
                bbox = None
                data_dir = self.data_dir
        # captions = self.captions[key]
        #embeddings = self.embeddings[index, :, :]

        img_name = '%s/train/all/%s' % (data_dir, key)
        #img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)


        flipResults = False
        if cfg.COND_TYPE == 'SIDE':
            # Create flipped image
             wrong_imgs = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm,
                            flip = True)

            #  As written, all "correct" images will be left image, so flip right/wrong with 50%
             if random.random() < 0.5:
                flipResults = True
                label = 1 - label

        else:

            wrong_ix = random.choice(self.dict_idx_by_class[1-label])
            wrong_key = self.filenames[wrong_ix]
            #wrong_ix = random.randint(0, len(self.filenames) - 1)
            #if(label == self.class_id[wrong_ix]):
            #    wrong_ix = random.randint(0, len(self.filenames) - 1)
            
            if self.bbox is not None:
                wrong_bbox = self.bbox[wrong_key]
            else:
                wrong_bbox = None
            wrong_img_name = '%s/train/all/%s' % (data_dir, wrong_key)
            #wrong_img_name = '%s/images/%s.jpg' % (data_dir, wrong_key)
            wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                                  wrong_bbox, self.transform, normalize=self.norm)                    
                
        #embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        #embedding = embeddings[embedding_ix, :]
        embedding = np.array([label]).astype(np.float32)
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        #print(index,"key: ",  key, "wrong key: ", wrong_key, label, embedding)
                
        if flipResults:
            return wrong_imgs,  imgs, embedding, key
        return imgs, wrong_imgs, embedding, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
