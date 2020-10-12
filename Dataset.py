import torch
import os
import random

import numpy as np
import torchvision.transforms.functional as TF
import xml.etree.ElementTree as Et

from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from cocoapi.PythonAPI.pycocotools.coco import COCO


def img_show(x):
    image, label = x['image'], x['target']
    if torch.is_tensor(image):
        image = TF.to_pil_image(image)
    image_draw = ImageDraw.Draw(image)
    boxes = label['boxes']
    for box in boxes:
        image_draw.rectangle([(box[0], box[1]), (box[2], box[3])])
    image.show()


class Crop_bd:

    def __init__(self, aspect=0.75):
        self.aspect = 0.75
    
    def __call__(self, x):
        image, label = x
        o_size = image.size  # [w, h]
        i = random.randint(0, int(o_size[0] * (1-self.aspect)))  # hor = w
        j = random.randint(0, int(o_size[1] * (1-self.aspect)))  # ver = h   
        image = TF.crop(image, j, i, o_size[1]*self.aspect, o_size[0]*self.aspect)
        boxes = label['boxes']
        idx = torch.ones((len(boxes)))

        for k, box in enumerate(boxes):

            box[0] -= i
            box[2] -= i
            box[1] -= j
            box[3] -= j

            if box[0] <= 0:
                box[0] = 1
            if box[1] <= 0:
                box[1] = 1
            if box[2] >= (o_size[0]*self.aspect):
                box[2] = o_size[0] * self.aspect-1
            if box[3] >= (o_size[1] * self.aspect):
                box[3] = o_size[1] * self.aspect-1
            if (box[2] <= 2) or (box[3] <= 2):
                idx[k] = 0
            x, y = (box[0] + box[2])/2, (box[1]+box[3])/2
            if x > o_size[0] * self.aspect or x < 0:
                idx[k] = 0
            if y >= o_size[1] * self.aspect or y < 0:
                idx[k] = 0

            w, h = box[2] - box[0], box[3] - box[1]
            if w < (o_size[0] * self.aspect * 0.03):
                idx[k] = 0
            if h < (o_size[1] * self.aspect * 0.03):
                idx[k] = 0
                
        idx = idx == 1
        label['boxes'] = boxes[idx]
        label['category'] = label['category'][idx]

        return (image, label)


class Resize_bd:
    def __init__(self, img_size, interpolation=Image.BILINEAR):
        self.interporation = interpolation
        self.img_size = img_size

    def __call__(self, x):
        image, label = x
        o_size = image.size
        scaling_w, scaling_h = self.img_size/o_size[0], self.img_size/o_size[1]
        image = TF.resize(image, (self.img_size, self.img_size), self.interporation)
        boxes = label['boxes']
        boxes[:, 0] *= scaling_w
        boxes[:, 2] *= scaling_w
        boxes[:, 1] *= scaling_h
        boxes[:, 3] *= scaling_h
        label['boxes'] = boxes

        return (image, label)


class VOCDataset:
    def __init__(self, img_size=416, train_mode=True):
        super(VOCDataset, self).__init__()
        
        if os.path.exists('./dataset/train.npy') and os.path.exists('./dataset/test.npy'):
            self.train_set = np.load('./dataset/train.npy')
            self.test_set = np.load('./dataset/test.npy')
            
        else:
            self.load_training_dataset()
        
        if train_mode:
            self.image_path = list(map(self.preprocessing_jpeg, self.train_set))
            self.label_path = list(map(self.prerpocessing_xml, self.train_set))
        else:
            self.image_path = list(map(self.preprocessing_jpeg, self.test_set))
            self.label_path = list(map(self.prerpocessing_xml, self.test_set))
        
        self.category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                         'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

        self.train_mode = train_mode
        self.resize_bd = Resize_bd(img_size=img_size)
        self.crop_bd = Crop_bd()
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose(
            [transforms.ColorJitter(brightness=0.75, hue=0.1, saturation=.75),
             transforms.ToTensor()]
        )
        self.va_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):

        img_name = self.image_path[idx]
        xml_name = self.label_path[idx]

        image = Image.open(img_name)
        xml = open(xml_name, 'r')
        tree = Et.parse(xml)
        root = tree.getroot()
        target = {}
        boxes = []
        labels = []
        objects = root.findall("object")

        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            
            name = obj.find("name").text
            label = self.category.index(name)
            labels.append(label)
        
        target['boxes'] = torch.tensor(boxes).float()
        target['category'] = torch.tensor(labels).long()
        if self.train_mode:
            image, target = self.crop_bd((image, target))
            image, target = self.resize_bd((image, target))
            image = self.transform(image)
        else:
            image, target = self.resize_bd((image, target))
            image = self.va_transform(image)

        return {'image': image, 'target': target}

    def preprocessing_jpeg(self, x):
        return './Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages/'+x+'.jpg'

    def prerpocessing_xml(self, x):
        return './Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/'+x+'.xml'

    def __len__(self):
        return len(self.image_path)
    
    def load_training_dataset(self):
        label_path = './Pascal_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
        validate_path = './Pascal_VOC_2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt'

        self.train_dataset = np.loadtxt(label_path, dtype='str')
        self.validate_dataset = np.loadtxt(validate_path, dtype='str')
        self.train_set = np.concatenate((self.validate_dataset, self.train_dataset), 
                                        axis=0)
        np.random.shuffle(self.train_set)

        self.train_set = self.train_set.tolist()
        length = len(self.train_set)
        self.test_set = []

        for i in range(int(0.2*length)):
            self.test_set.append(self.train_set.pop())
        
        self.train_set = np.array(self.train_set)
        self.test_set = np.array(self.test_set)

        np.save('./dataset/train.npy', self.train_set)
        np.save('./dataset/test.npy', self.test_set)


class ImageNetDataset(Dataset):
    
    def __init__(self, img_size=256, val_mode=False):
        super(ImageNetDataset, self).__init__()
        
        self.cur_path = '/home/mmc-server3/Server/server2/seungju/YOLO/'
        self.data_path = '/home/mmc-server3/Server/dataset/ILSVRC2012_img_train/'
        if val_mode:
            self.data_path = '/home/mmc-server3/Server/dataset/ILSVRC2012_img_val/'
            self.val_label = np.loadtxt('./dataset/ILSVRC2011_validation_ground_truth.txt')
            self.cat = np.empty((0), dtype='str')
        
        data = np.empty((0,), dtype='str')
        self.category = os.listdir(self.data_path)

        self.category.sort()

        for i in self.category:
            path = self.data_path + i
            k = os.listdir(path)
            k.sort()
            k = np.array(k)
            k = k.reshape((-1,))
            data = np.concatenate((data, k), 0)
            if val_mode:
                temp = np.stack([i for j in range(len(k))])
                
                self.cat = np.concatenate((self.cat, temp), 0)
        self.data = data
        print("Total Image : {} // Total Cateogry : {}".format(self.data.shape[0],
              len(self.category)))
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        # crop = transforms.RandomCrop(int(img_size))
        colojiter = transforms.ColorJitter(brightness=.75, saturation=.75, hue=.1)
        
        self.transformation = transforms.Compose([
            transforms.RandomResizedCrop(img_size), 
            transforms.RandomHorizontalFlip(), 
            colojiter, transforms.ToTensor()])
        self.va_transformation = transforms.Compose([
            transforms.Resize([img_size, img_size]), 
            transforms.ToTensor()])
        self.val_mode = val_mode
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = self.data[idx]
        if self.val_mode:
            cat = self.cat[idx]
            cat_idx = self.category.index(cat)
        else:
            cat = data[:9]
            cat_idx = self.category.index(cat)
        path = self.data_path + cat + '/' + data
        image = Image.open(path)
        image = image.convert('RGB')
        
        if self.val_mode:
            image = self.va_transformation(image)
        else:
            image = self.transformation(image)
        
        return (image, cat_idx)


class anchor_box:
    def __init__(self, path='./dataset/train.npy', k=5):

        self.k = k
        path = np.load(path)
        path = list(map(self.prerpocessing_xml, path))
        box = []
        for xml_name in path:
            xml = open(xml_name, 'r')
            tree = Et.parse(xml)
            root = tree.getroot()
            objects = root.findall("object")
            size = root.find("size")
            w = float(size.find("width").text)
            h = float(size.find("height").text)
            for obj in objects:
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                xmax = float(bndbox.find("xmax").text)
                ymin = float(bndbox.find("ymin").text)
                ymax = float(bndbox.find("ymax").text)
                box.append([(xmax-xmin)/w, (ymax-ymin)/h])

        # if isinstance(box, np.ndarray):
        #     pass
        # else:
        #     box = np.array(box, dtype=np.float32)
        
        # if box.shape[1] !=2:
        #     box = self.prerpocess_box(box)
        self.box = box
        self.cand = np.array(random.sample(self.box, k))
        self.box = np.array(self.box, dtype=np.float32)
        self.idx = np.empty((len(self.box)))

    def prerpocessing_xml(self, x):
        return './Pascal_VOC_2012/VOCdevkit/VOC2012/Annotations/'+x+'.xml'

    def prerpocess_box(self, box):
        new_box = np.zeros((len(self.box), 2))
        new_box[:, 0] = box[:, 2] - box[:, 0]
        new_box[:, 1] = box[:, 3] - box[:, 1]
        return new_box
    
    def run(self):
        for i in range(20):
            self.calculate_idx()
            self.calculate_cand_box()
        self.sort()
        
        print("----------- anchor_box -----------")
        print(self.cand[::-1])
        np.save('./dataset/anchor_box.npy', self.cand[::-1])
    
    def sort(self):
        
        temp = self.cand.copy()
        for i in range(self.k):
            area = self.cand[:, 0] * self.cand[:, 1]
            idx = np.argmax(area)
            temp[i] = self.cand[idx]
            area = np.delete(area, idx)
            self.cand = np.delete(self.cand, idx, axis=0)
        self.cand = temp

    def calculate_cand_box(self):
        for i in range(self.k):
            idx = self.idx == i
            box = self.box[idx]
            cand_box = np.mean(box, axis=0)
            self.cand[i, :] = cand_box

    def calculate_idx(self):
        for i, box in enumerate(self.box):
            idx = self.caclulate_iou(box)
            self.idx[i] = idx

    def caclulate_iou(self, box):
        w_min = np.minimum(box[0], self.cand[:, 0])
        h_min = np.minimum(box[1], self.cand[:, 1])

        area = w_min * h_min

        area_box = box[0] * box[1]
        cand_area = self.cand[:, 0] * self.cand[:, 1]

        ious = area/(area_box + cand_area-area)

        idx = np.argmax(ious)

        return idx


class cocoDataSet(Dataset):

    def __init__(self, iSize=416, train_mode=True):
        super(cocoDataSet, self).__init__()
        self.iSize = iSize
        self.data_folder = './MS_COCO_2017'
        self.trainMode = train_mode
        if train_mode:
            self.dataType = 'train2017'
        else:
            self.dataType = 'val2017'
        self.annFile = './MS_COCO_2017_Anno/annotations/instances_{}.json'.format(self.dataType)
        self.coco = COCO(self.annFile)
        self.Idx = list(self.coco.imgs.keys())
        self.catIdx = list(self.coco.cats.keys())

        self.resize_bd = Resize_bd(img_size=iSize)
        self.crop_bd = Crop_bd()
        self.transform = transforms.Compose(
            [transforms.ColorJitter(brightness=0.75, hue=0.1, saturation=.75),
             transforms.ToTensor()]
        )
        self.va_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_image(self, idx):
        iname = '{0:0>12}'.format(idx)+'.jpg'
        path = os.path.join(self.data_folder, self.dataType, iname)
        image = Image.open(path).convert("RGB")
        return image
    
    def __len__(self):
        return len(self.coco.imgs.keys())
        
    def __getitem__(self, idx):
        target = {}
        idx = self.Idx[idx]
        image = self.load_image(idx)
        anns = self.coco.imgToAnns[idx]

        bboxes = []
        cats = []

        for ann in anns:
            x, y, w, h = [i for i in ann['bbox']]
            box = [x, y,
                   x + w, y + h]
            
            bboxes.append(box)
            cats.append(self.catIdx.index(ann['category_id']))
        target['boxes'] = torch.tensor(bboxes)
        target['category'] = torch.tensor(cats)

        if self.trainMode:
            image, target = self.crop_bd((image, target))
            image, target = self.resize_bd((image, target))
            prob = np.random.rand()
            if prob > 0.7:
                image = self.transform(image)
            else:
                image = self.va_transform(image)
        else:
            image, target = self.resize_bd((image, target))
            image = self.va_transform(image)

        return {'image': image, 'target': target}


if __name__ == "__main__":
    dataset = VOCDataset()
    b = dataset[10]
    a = np.random.randint(0, 1000, 20)
    # for i in a:
    #     test = dataset[i]
    #     img_show(test)
    # anchor = anchor_box()
    # anchor.run()
    cocoDa = cocoDataSet(train_mode=False)
    for i in a:
        test = cocoDa[i]
        img_show(test)
    
