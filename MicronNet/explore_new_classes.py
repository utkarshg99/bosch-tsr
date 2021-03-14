from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from data import initialize_data # data.py in the same folder
from model import Net

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='DatasetAug', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='explored.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()

state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()

from data import test_data_transforms

test_dir = args.data + '/test_images'

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(os.listdir(test_dir)):
    if 'png' in f:
        data = test_data_transforms(image=np.array(pil_loader(test_dir + '/' + f)))["image"]
        data = data.float()
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        data = Variable(data)
        model.eval()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        # print(np.sum(np.exp(output.detach().numpy())))

        file_id = f[0:len(f)-4]
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + args.outfile)