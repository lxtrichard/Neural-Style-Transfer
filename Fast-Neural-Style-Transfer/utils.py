import os
from PIL import Image
import cv2
import glob
from collections import namedtuple

import torch
from torchvision import models,transforms



class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_2 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        for i in range(4):
            self.relu1_2.add_module(str(i), features[i])
        for i in range(4,9):
            self.relu2_2.add_module(str(i), features[i])
        for i in range(9,16):
            self.relu3_2.add_module(str(i), features[i])
        for i in range(16,23):
            self.relu4_2.add_module(str(i), features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1_2 = self.relu1_2(X)
        h_relu2_2 = self.relu2_2(h_relu1_2)
        h_relu3_2 = self.relu3_2(h_relu2_2)
        h_relu4_2 = self.relu4_2(h_relu3_2)
        vgg_out = namedtuple("vgg_out",["relu1_2","relu2_2","relu3_2","relu4_2"])
        out = vgg_out(h_relu1_2,h_relu2_2,h_relu3_2,h_relu4_2)
        return out

def load_img(img_path):
    img = Image.open(img_path)
    return img

def save_img(img_path,img):
    img = img.clone().numpy()
    img = (img.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(img_path)

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return height, width, fps

def get_frame(video_path,frame_path):
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    success = True
    count=1
    while success:
        cv2.imwrite(os.path.join(frame_path,"{:03d}.jpg".format(count)), img)
        success, img = cap.read()
        count+=1
    print("Done getting all frames.")

def make_video(frame_path, save_name, height, width,fps):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(save_name, fourcc, fps, (width,height))
    frame_name = sorted(glob.glob(os.path.join(frame_path,"*.jpg")))
    frame_set = [cv2.imread(frame_name[i]) for i in range(len(frame_name))]
    for frame in frame_set:
        out.write(frame)
    out.release()
    print("Done writing video.")



