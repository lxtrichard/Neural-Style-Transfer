import os
import time
import argparse
from PIL import Image

import torch
import torchvision
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils
from model import Model
from utils import LoadDataset

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_dir(args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

def test_dir(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    train_dataset = LoadDataset(args.content_dir, args.style_dir,transform=transform,shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = LoadDataset(args.content_dir, args.style_dir,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_iter = iter(test_loader)

    model = Model().to(device)
    optimizer = Adam(model.parameters(),lr=args.learning_rate)

    print("Begin training")
    start_time = time.time()
    T_loss = []
    for epoch in range(args.epochs):
        count = 0
        for ids, (content,style) in enumerate(train_loader):
            n_batch = len(content)
            count += n_batch
            content = content.to(device)
            style = style.to(device)
            loss = model(content,style)
            total_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            T_loss.append(total_loss)
            if (ids+1) % args.log_interval == 0:
                print("Epoch {}\t [{}/{}]\t loss:{:.5f}".format(
                    epoch+1,count,len(train_dataset),
                    total_loss
                ))
                content, style = next(test_iter)
                content = content.to(device)
                style = style.to(device)
                with torch.no_grad():
                    out = model.generate(content, style)
                content = utils.denorm(content, device)
                style = utils.denorm(style, device)
                out = utils.denorm(out, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                save_image(res, './check/epoch_{}_iter_{}.jpg'.format(epoch,ids+1), nrow=args.batch_size)
        model_name = "epoch_{}.model".format(epoch)
        model_path = os.path.join(args.model_dir, model_name)
        torch.save(model.state_dict(), model_path)
    plt.plot(range(len(T_loss)), T_loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.savefig('./results/loss/T_loss.png')
    print("Done! The total time is {}".format(time.time()-start_time))

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    model = Model()
    if args.model_dir is not None:
        model_dict = torch.load(args.model_dir, map_location='cpu')
        model.load_state_dict(model_dict)
    model = model.to(device)

    content = Image.open(args.content)
    style = Image.open(args.style)
    c_tensor = transform(content).unsqueeze(0).to(device)
    s_tensor = transform(style).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor,s_tensor,args.alpha)
    out = utils.denorm(out,device)
    c_name, s_name = args.content.split("/")[-1].split(".")[0], args.style.split("/")[-1].split(".")[0]

    if args.alpha==1:
        output_name = os.path.join(args.output_dir,"{}_{}.jpg".format(c_name, s_name))
    else:
        output_name = os.path.join(args.output_dir,"{}_{}_{}.jpg".format(c_name, s_name,args.alpha))
    save_image(out,output_name,nrow=1)
    

    w,h = content.size
    background = Image.new("RGB",(w*2,h))
    out = Image.open(output_name).resize((w,h))
    out_style = style.resize((w//4,h//4))
    background.paste(content,(0,0))
    background.paste(out,(w,0))
    background.paste(out_style,(w,0))
    if args.alpha==1:
        output_name = os.path.join(args.output_dir,"comb_{}_{}.jpg".format(c_name, s_name))
    else:
        output_name = os.path.join(args.output_dir,"comb_{}_{}_{}.jpg".format(c_name, s_name, args.alpha))
    background.save(output_name)

    out.paste(out_style,(0,0))
    out.save(os.path.join(args.output_dir,"ws_{}_{}.jpg".format(c_name, s_name)))

    print("results saved at {}".format(args.output_dir))

    

def main():
    main_args = argparse.ArgumentParser()
    sub_args = main_args.add_subparsers(title="mode",dest="mode")
    train_args = sub_args.add_parser("train")
    test_args = sub_args.add_parser("test")

    train_args.add_argument("--model_dir",type=str,default="./models")
    train_args.add_argument("--img_size",type=int,default=512)
    train_args.add_argument("--content_dir",type=str,default="./data/content")
    train_args.add_argument("--style_dir",type=str,default="./data/style")
    train_args.add_argument("--batch_size",type=int,default=8)
    train_args.add_argument("--learning_rate","--lr",type=float,default=5e-5)
    train_args.add_argument("--epochs",type=int,default=1)
    train_args.add_argument("--log_interval", type=int,default=500)


    test_args.add_argument("--model_dir",type=str,default="./models/AdaIN.model")
    test_args.add_argument("--img_size",type=int,default=512)
    test_args.add_argument("--content","-c",type=str,default="./img/content.jpg")
    test_args.add_argument("--style","-s",type=str,default="./img/style.jpg")
    test_args.add_argument("--alpha",type=float,default=1)
    test_args.add_argument("--output_dir",type=str,default="./results")


    args = main_args.parse_args()
    if args.mode == "train":
        train_dir(args)
        train(args)
    elif args.mode == "test":
        test_dir(args)
        test(args)

    




if __name__ == "__main__":
    main()