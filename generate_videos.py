import torch
from torchvision.transforms import functional as tr
import datasets.transforms as tr_custom
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
import skvideo.io
import cv2
import os

import configuration as conf

input_video_paths = ['video_tests/input/Suji.MOV', 'video_tests/input/Vicky_1.MOV', 'video_tests/input/Vicky_2.MOV',
                     'video_tests/input/Vicky_3.MOV']
output_dir = 'video_tests/output'

_, _, _, model, _, _ = conf.params()
model.eval()

input_video_names = [p.split('/')[-1].split('.')[0] for p in input_video_paths]
output_video_paths = [os.path.join(output_dir, n + '.mp4') for n in input_video_names]

for input_video, output_video in zip(input_video_paths, output_video_paths):
    fr = cv2.VideoCapture(input_video).get(cv2.CAP_PROP_FPS)
    input_video = skvideo.io.vread(input_video)
    output_video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fr,
                                   (conf.fullres[1], conf.fullres[0]))

    avg_time_per_frame = 0
    for input_frame in tqdm(input_video):
        if input_frame is None:
            break

        input_frame = Image.fromarray(input_frame)
        input_frame = tr_custom.center_crop([input_frame], aspect_ratio=conf.fullres[1]/conf.fullres[0])[0]
        input_frame_fullres = tr.resize(input_frame, conf.fullres)
        input_frame_lowres = tr.resize(input_frame_fullres, conf.lowres, interpolation=Image.NEAREST)
        input_frame_fullres = tr.to_tensor(input_frame_fullres)
        input_frame_lowres = tr.to_tensor(input_frame_lowres)

        if torch.cuda.is_available():
            input_frame_fullres, input_frame_lowres = input_frame_fullres.cuda(), input_frame_lowres.cuda()
        input_frame_fullres, input_frame_lowres = input_frame_fullres.unsqueeze(0), input_frame_lowres.unsqueeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            output_frame = model(input_frame_lowres, input_frame_fullres)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        output_frame = output_frame.squeeze(0).cpu()
        avg_time_per_frame += (end_time - start_time) / len(input_video)

        output_frame = tr.to_pil_image(output_frame)
        output_frame = np.asarray(output_frame)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        output_video.write(output_frame)

    print(avg_time_per_frame)

    output_video.release()
