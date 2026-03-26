import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image


def mysort(dataset_path, in1k=False):
    types = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp','*.tif']
    dataset_img_path = []
    for type in types:
        dataset_img_path.extend(glob.glob(os.path.join(dataset_path, type)))
    if not in1k:
        try:
            dataset_img_path = sorted(dataset_img_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        except Exception as e:
            print(f"The name of images in {dataset_path} is not numerical!")
    return dataset_img_path


def pil_loader(path: str, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            # print('use gray')
            return img.convert('RGB').convert('L').convert('RGB')
        return img.convert('RGB')

import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

class PairedTransform:
    """
    同步对 RGB 和 Thermal 图像执行相同的 transform。
    等价于：
    Compose([
        RandomResizedCrop((224,224), scale=(0.2,1.0), ratio=(0.75,1.3333), interpolation=BICUBIC),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=[0.3801,0.3801,0.3801], std=[0.1871,0.1871,0.1871])
    ])
    """
    def __init__(self, size=(224, 224), scale=(0.2, 1.0), ratio=(0.75, 1.3333),
                 hflip_prob=0.5, mean=[0.3801]*3, std=[0.1871]*3):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

    def __call__(self, img_rgb: Image.Image, img_ir: Image.Image):
        # 1️⃣ 同步随机裁剪参数
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img_rgb, scale=self.scale, ratio=self.ratio
        )
        img_rgb = F.resized_crop(img_rgb, i, j, h, w, self.size, interpolation=Image.BICUBIC)
        img_ir = F.resized_crop(img_ir, i, j, h, w, self.size, interpolation=Image.BICUBIC)

        # 2️⃣ 同步随机翻转
        if random.random() < self.hflip_prob:
            img_rgb = F.hflip(img_rgb)
            img_ir = F.hflip(img_ir)

        # 3️⃣ ToTensor + Normalize
        img_rgb = F.to_tensor(img_rgb)
        img_ir = F.to_tensor(img_ir)

        img_rgb = F.normalize(img_rgb, mean=self.mean, std=self.std)
        img_ir = F.normalize(img_ir, mean=self.mean, std=self.std)

        return img_rgb, img_ir


class InfMix(Dataset):

    def __init__(self, infpre_path, in1k_path, coco_path, transforms=None, spec_dataset=None, use_in1k=False, per_cls_num=200,
                 use_coco=False, data_ratio=1.0, rgb_gray=False):
        self.transforms = PairedTransform(size=(224, 224))#transforms
        self.use_in1k = use_in1k
        self.rgb_gray = rgb_gray
        self.use_coco = use_coco
        self.spec_dataset = spec_dataset
        self.per_cls_num = per_cls_num
        self.img_list = self._make_dataset(infpre_path, in1k_path, coco_path)
        if data_ratio != 1.0:
            total_data_num = len(self.img_list)
            data_num = int(total_data_num * data_ratio)
            data_sample_interval = int(1 / data_ratio)
            self.img_list = self.img_list[::data_sample_interval]
    
        
    def _make_dataset(self, infpre_path, in1k_path, coco_path):
        """
        infpre_path/
            dataset1_name/
                thermal/
                    image1.jpg
                    image2.jpg
                    ...
                rgb/
                    image1.jpg
                    image2.jpg
            dataset2_name/
                thermal/
                    ...
                rgb/
                    ...
        """
        thermal_img_list = []
        rgb_img_list = []#1015
        if self.spec_dataset is not None:
            datasets = self.spec_dataset
        else:
            datasets = os.listdir(infpre_path)
        
        for dataset in datasets:
            dataset_path = os.path.join(infpre_path, dataset)
            if not os.path.isdir(dataset_path):
                continue
            thermal_path = os.path.join(dataset_path, r'thermal')
            if not os.path.exists(thermal_path):
                continue
            dataset_thermal_img_list = mysort(thermal_path)
            thermal_img_list.extend(dataset_thermal_img_list)

            # rgb 10.12
            rgb_path = os.path.join(dataset_path, r'rgb')
            if not os.path.exists(rgb_path):
                continue
            dataset_rgb_img_list = mysort(rgb_path)
            thermal_img_list.extend(dataset_rgb_img_list)#1015

        if self.use_in1k:
            for sub_dataset in os.listdir(in1k_path):
                dataset_thermal_img_list = mysort(os.path.join(in1k_path, sub_dataset), in1k=True)
                if self.per_cls_num != 0:
                    dataset_thermal_img_list = dataset_thermal_img_list[:self.per_cls_num]
                thermal_img_list.extend(dataset_thermal_img_list)
                
        if self.use_coco:
            sub_dataset_list = ['train2017']
            for sub_dataset in sub_dataset_list:
                dataset_thermal_img_list = mysort(os.path.join(coco_path, sub_dataset))
                thermal_img_list.extend(dataset_thermal_img_list)  
                
        return thermal_img_list
    
    
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        try:#1208
            if '/rgb/' in img_path:
                img = pil_loader(img_path)  # , gray=self.rgb_gray)
                img2_path = img_path.replace('/rgb/', '/thermal/')
                if os.path.exists(img2_path):
                    img2 = pil_loader(img2_path)  # , gray=self.rgb_gray)
                else:
                    img2  =img
            elif '/thermal/' in img_path:
                img = pil_loader(img_path)  # , gray=self.rgb_gray)
                img2_path = img_path.replace('/thermal/', '/rgb/')
                if os.path.exists(img2_path):
                    img2 = pil_loader(img2_path)  # , gray=self.rgb_gray)
                else:
                    img2 = img
            else:
                img = pil_loader(img_path)
                img2 = img
        except Exception as e:
            print(f"[FATAL ERROR] index {index} failed: {e}")
            print(img_path,img2_path)

        # img_ir_path = img_path.replace('/rgb/', '/thermal/')
        # img_rgb_path = img_path.replace('/thermal/', '/rgb/')
        #
        # if os.path.exists(img_ir_path):
        #     img2 = pil_loader(img_ir_path)  # , gray=self.rgb_gray)
        # elif os.path.exists(img_rgb_path):
        #     img2 = pil_loader(img_rgb_path)
        # else:
        #     img2 = img

        if self.transforms is not None:
            img, img2 = self.transforms(img, img2)

        return [img,img2]
        
    

if __name__ == '__main__':
    transform_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.3801, 0.3801, 0.3801], std=[0.1871, 0.1871, 0.1871]),
        ])
    infpre_path = r'path/to/infpre'
    in1k_path = r'path/to/imagenet'
    coco_path = r'path/to/coco'
    dataset_train = InfMix(infpre_path=infpre_path, in1k_path=in1k_path, coco_path=coco_path, transforms=transform_train, use_in1k=True, use_coco=True, per_cls_num=200, rgb_gray=True)
    print(len(dataset_train))
