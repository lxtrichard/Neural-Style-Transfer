import os
import glob
import argparse
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

from image_transform_net import ImageTransformNet
import utils
from utils import Vgg16

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_dir(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.checkpoint_dir is not None and not (os.path.exists(args.checkpoint_dir)):
        os.makedirs(args.checkpoint_dir)

def video_dir(args):
    if args.frame_path is not None and not (os.path.exists(args.frame_path)):
        os.makedirs(args.frame_path)
    if args.style_path is not None and not (os.path.exists(args.style_path)):
        os .makedirs(args.style_path)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform=transform)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size)
    transform_net = ImageTransformNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transform_net.parameters(),lr=args.lr)
    
    style = utils.load_img(args.style_img)
    style = style_transform(style)
    style = style.repeat(args.batch_size,1,1,1).to(device)

    style_features = vgg(style)
    style_gram = [utils.gram(x) for x in style_features]

    style_name = args.style_img.split("/")[-1].split(".")[0]
    print("Begin Training {}".format(style_name))
    T_loss = []
    for epoch in range(args.epochs):
        transform_net.train()
        C_loss = 0
        S_loss = 0
        count = 0
        for ids, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transform_net(x)

            x_features = vgg(x)
            y_features = vgg(y)

            style_loss = 0
            for y_ft, s_gm in zip(y_features,style_gram):
                y_gm = utils.gram(y_ft)
                style_loss += mse_loss(y_gm,s_gm[:n_batch,:,:])
            style_loss = args.beta * style_loss
            content_loss = args.alpha * mse_loss(x_features.relu2_2, y_features.relu2_2)

            total_loss = content_loss + style_loss
            T_loss.append(total_loss)
            total_loss.backward()
            optimizer.step()

            C_loss = content_loss.item()
            S_loss = style_loss.item()

            if (ids+1) % args.log_interval == 0:
                print("Epoch {}\t [{}/{}]\t content:{:.5f}\t style:{:.5f}\t total:{:.5f}".format(
                    epoch+1,count,len(train_dataset),
                    C_loss/(ids+1),S_loss/(ids+1),
                    (C_loss+S_loss)/(ids+1)
                ))
    plt.figure(1)
    plt.plot(range(len(T_loss)), T_loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Total loss')
    plt.savefig('T_loss.png')
    
    transform_net.eval().cpu()
    model_name = "{}.model".format(style_name)
    model_path = os.path.join(args.model_dir,model_name)
    torch.save(transform_net.state_dict(),model_path)
    print("Done!")

def stylize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img.mul(255))
    ])
    content_img = utils.load_img(args.content_path)
    content_img = content_transform(content_img)
    content_img = content_img.unsqueeze(0).to(device)
    with torch.no_grad():
        model = ImageTransformNet()
        model_dict = torch.load(args.model)
        model.load_state_dict(model_dict)
        model.to(device)
        output = model(content_img).cpu()
    content_name = args.content_path.split("/")[-1].split(".")[0]
    style_name = args.model.split("/")[-1].split(".")[0]
    output_path = os.path.join("./results",content_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    utils.save_img(os.path.join(output_path,"{}.jpg".format(style_name)), output[0])

def stylize_video(args):
    h,w,fps = utils.video_info(args.video_path)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # utils.get_frame(args.video_path,args.frame_path)
    if args.frame_path != None:
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.mul(255))
        ])
        frame_names = sorted(glob.glob(os.path.join(args.frame_path,"*.jpg")))
        frame_set = [utils.load_img(x) for x in frame_names]
        with torch.no_grad():
            model = ImageTransformNet()
            model_dict = torch.load(args.model)
            model.load_state_dict(model_dict)
            model.to(device)
            count = 1
            for i in range(len(frame_set)):
                content_img = frame_set[i]
                content_img = content_transform(content_img)
                content_img = content_img.unsqueeze(0).to(device)
                output = model(content_img).cpu()
                utils.save_img(os.path.join(args.style_path,"{:03d}.jpg".format(count)), output[0])
                count += 1
    style_name = args.model.split("/")[-1].split(".")[0]
    content_name = args.video_path.split("/")[-1].split(".")[0]
    save_name = "./video/{}_{}.mp4".format(content_name, style_name)
    utils.make_video(args.style_path,save_name,int(h),int(w),fps)
    


def main():
    main_args = argparse.ArgumentParser()
    sub_args = main_args.add_subparsers(title="mode",dest="mode")
    train_args = sub_args.add_parser("train")
    style_args = sub_args.add_parser("style")
    video_args = sub_args.add_parser("video")

    train_args.add_argument("--epochs", type=int,default=2)
    train_args.add_argument("--batch-size",type=int,default=4)
    train_args.add_argument("--dataset",type=str,default="./data/")
    train_args.add_argument("--style_img",type=str,default="./style_img/mosaic.jpg")
    train_args.add_argument("--model_dir",type=str,default="./models")
    train_args.add_argument("--checkpoint_dir",type=str,default=None)
    train_args.add_argument("--img_size",type=int,default=256)
    train_args.add_argument("--alpha",type=float,default=1e5)
    train_args.add_argument("--beta",type=float,default=1e10)
    train_args.add_argument("--lr",type=float,default=1e-3)
    train_args.add_argument("--log_interval",type=int,default=500)
    
    style_args.add_argument("--content_path",type=str,required=True)
    style_args.add_argument("--model",type=str,default="./models/mosaic.model")

    video_args.add_argument("--video_path",type=str,default="./video/ballet.mp4")
    video_args.add_argument("--frame_path",type=str,default="./video/raw_frame")
    video_args.add_argument("--style_path",type=str,default="./video/style_frame")
    video_args.add_argument("--model",type=str,default="./models/mosaic.model")

    args = main_args.parse_args()
    if args.mode == "train":
        train_dir(args)
        train(args)
    elif args.mode == "style":
        stylize(args)
    elif args.mode == "video":
        video_dir(args)
        stylize_video(args)

if __name__ == "__main__":
    main()

