from torch.utils.data import Dataset
import torchvision.transforms.functional as tr
from PIL import Image
import os
import datasets.transforms as tr_custom


class BaseDataset(Dataset):

    def __init__(self, data_dir, lowres, fullres, training=False):
        super().__init__()
        assert fullres[0] / fullres[1] == lowres[0] / lowres[1]
        self.training = training
        self.lowres, self.fullres = lowres, fullres
        self.aspect_ratio = fullres[1] / fullres[0]
        self.data = self.read_data(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image, target = self.data[item]['image'], self.data[item]['target']
        image, target = Image.open(image).convert('RGB'), Image.open(target).convert('RGB')
        image_lowres, image_fullres, target = self.transform(image, target)
        return image_lowres, image_fullres, target

    def transform(self, image, target):
        if self.training:
            image, target = tr_custom.random_horizontal_flip([image, target])
            image, target = tr_custom.random_rotation([image, target], angle=15)
            image, target = tr_custom.random_crop([image, target], scale=(0.8, 1.0),
                                                  aspect_ratio=self.aspect_ratio)
        else:
            image, target = tr_custom.center_crop([image, target], aspect_ratio=self.aspect_ratio)
        image_fullres, target = [tr.resize(img, self.fullres) for img in [image, target]]
        image_lowres = tr.resize(image_fullres, self.lowres, interpolation=Image.NEAREST)
        image_lowres, image_fullres, target = [tr.to_tensor(img) for img in [image_lowres, image_fullres, target]]
        return image_lowres, image_fullres, target

    def read_data(self, data_dir):
        data = os.listdir(data_dir)
        data = [d for d in data if '_target' in d]
        data = [os.path.join(data_dir, d) for d in data]
        data = [{'image': d.replace('_target', ''), 'target': d} for d in data]
        return data
