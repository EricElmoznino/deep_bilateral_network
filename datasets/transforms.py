import torchvision.transforms.functional as tr
from PIL import Image
import random


def random_horizontal_flip(images, probability=0.5):
    if random.random() < probability:
        images = [tr.hflip(img) for img in images]
    return images


def random_crop(images, scale, aspect_ratio=None):
    crop_ratio = random.uniform(scale[0], scale[1])
    w, h = images[0].size
    if aspect_ratio is None:
        tw, th = int(crop_ratio * w), int(crop_ratio * h)
    else:
        if min(w * aspect_ratio, h) == h:
            tw, th = int(crop_ratio * h / aspect_ratio), int(crop_ratio * h)
        else:
            tw, th = int(crop_ratio * w), int(crop_ratio * w * aspect_ratio)
    if w == tw and h == th:
        return images
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    images = [tr.crop(img, i, j, th, tw) for img in images]
    return images


def random_rotation(images, angle):
    angle = random.uniform(-angle, angle)
    images = [tr.rotate(img, angle, False, False, None) for img in images]
    return images


def center_crop(images, aspect_ratio=1.0):
    w, h = images[0].size
    if min(w, h) == w:
        tw, th = w, int(w * aspect_ratio)
    else:
        tw, th = int(h / aspect_ratio), h
    if w == tw and h == th:
        return images
    images = [tr.center_crop(img, output_size=(th, tw)) for img in images]
    return images
