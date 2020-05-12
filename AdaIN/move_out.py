import glob
import os
import shutil
import skimage.io as skio
import concurrent.futures

from PIL import Image,ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def mv_content(img_path):
    img = Image.open(img_path)
    content_dest = "./data/tmp/content/"
    if len(img.split())!=3:
        print(img_path)
        shutil.move(img_path,content_dest)

def mv_style(img_path):
    img = Image.open(img_path)
    style_dest = "./data/tmp/style/"
    if len(img.split())!=3:
        print(img_path)
        shutil.move(img_path,style_dest)      

content_imgs = glob.glob(os.path.join("./data/content","*"))
style_imgs = glob.glob(os.path.join("./data/style","*"))
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    executor.map(mv_content, content_imgs)
    executor.map(mv_style, style_imgs)


