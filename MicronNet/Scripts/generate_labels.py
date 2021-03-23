import cv2
import albumentations as A
import os
import PIL.Image as Image
import numpy as np

src_dir = "DatasetDiff2/val_images"
dest_dir = "DatasetTesting/"
nclasses = 48
fx = []
base = 13081
curr = 1

# def transfer(basetr, transtr): 
for i in range(nclasses):
    fldr=("/0000" if i<10 else "/000")+str(i)
    for path in os.listdir(src_dir+fldr):
        full_path = os.path.join(src_dir+fldr, path)
        strng = str(base+curr)+","+str(i)
        curr = curr+1
        fx.append(strng)
        os.rename(full_path, dest_dir + '/' + str(base+curr-1) + ".png")

fx = "\n".join(fx)
output_file = open("testresults.csv", "w")
output_file.write(fx)
