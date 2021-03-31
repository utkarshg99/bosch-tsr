from flask import Flask, request, render_template
from flask_cors import CORS
from flask_pymongo import PyMongo
import os
import math
import uuid
import shutil
import pickle
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import redis
from rq import Queue

from utility import pil_loader, readb64, writeb64, reset_stats, reset_eval_stats, augment,\
                    generateImgFilePaths, generateImgFilePathsInc, scan_dir, stats, eval_stats
from dataset import transform, data_transforms, train_data_transforms, ADataset
from model import Net, ExtendNet, freeze_params, extended_cost, FNet
from segregate import random_segregate, random_segregate_all, smart_segregate, tsne_plot
from heatmap import generate_heat_map, anchor_heat_map

# Folder paths
main_train = os.path.join(os.path.dirname(__file__), 'static/Main/Train/')
main_test = os.path.join(os.path.dirname(__file__), 'static/Main/Test/')
main_val = os.path.join(os.path.dirname(__file__), 'static/Main/Val/')
additional_temp = os.path.join(os.path.dirname(__file__), 'static/Additional/temp/')
additional_train = os.path.join(os.path.dirname(__file__), 'static/Additional/Train/')
additional_test = os.path.join(os.path.dirname(__file__), 'static/Additional/Test/')
additional_val = os.path.join(os.path.dirname(__file__), 'static/Additional/Val/')
base_48_train = os.path.join(os.path.dirname(__file__), 'static/Base_48/Train/')
base_48_test = os.path.join(os.path.dirname(__file__), 'static/Base_48/Test/')
base_48_val = os.path.join(os.path.dirname(__file__), 'static/Base_48/Val/')
temp = os.path.join(os.path.dirname(__file__), 'temp/')

# Initialize Flask
app = Flask(__name__, static_url_path='/static', template_folder='static')
CORS(app)

# MongoDB setup
app.config['MONGO_DBNAME'] = 'interiit'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/interiit'
mongo = PyMongo(app)

# Redis setup
r = redis.Redis()
q = Queue(connection=r)

# For managing rq jobs
job_id = [None, None, None, None, None]

# Create folders if not available
if not os.path.exists("temp"):
    os.mkdir("temp")
if not os.path.exists("generated"):
    os.mkdir("generated")
if not os.path.exists("static/Additional"):
    os.mkdir("static/Additional")
    os.mkdir("static/Additional/temp")
    os.mkdir("static/Additional/Test")
    os.mkdir("static/Additional/Train")
    os.mkdir("static/Additional/Val")
    for i in range(48):
        os.mkdir("static/Additional/Val/" + str(i))
        os.mkdir("static/Additional/Test/" + str(i))
        os.mkdir("static/Additional/Train/" + str(i))


# Create stats files if not available
if not os.path.exists('stats.bin'):
    with open('stats.bin', 'wb') as f:
        pickle.dump(stats, f)
for i in range(5):
    if not os.path.exists('eval_stats' + str(i) + '.bin'):
        with open('eval_stats' + str(i) + '.bin', 'wb') as f:
            pickle.dump(eval_stats, f)


# Model Inference
def inference(model, path):
    data = data_transforms(image=np.array(pil_loader(path)))["image"]
    data = data.float()
    data = data.view(1, data.size(0), data.size(1), data.size(2))
    data = Variable(data)
    with torch.no_grad():
        output = model(data)
        pred = output.data.max(1, keepdim=True)
    return pred


# Model Training
def train_model(num_classes, batch_size, epochs, lr, momentum, decay, step, l2_norm, name, thres_acc,
                thres_epoch, datas, prev_classes, prev_name, freeze, aug):
    torch.manual_seed(1)
    log_interval = 180

    if aug:
        trans = train_data_transforms
    else:
        trans = data_transforms

    # If GTSRB
    if not datas:
        fxc = ADataset(images_filepaths=generateImgFilePaths(base_48_train, True), transform=trans)
        train_loader = torch.utils.data.DataLoader(fxc, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

        fxv = ADataset(images_filepaths=generateImgFilePaths(base_48_val, True), transform=trans)
        val_loader = torch.utils.data.DataLoader(fxv, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)
    # If GTSRB_48
    elif datas == "base":
        fxc = ADataset(images_filepaths=generateImgFilePaths(base_48_train, False), transform=trans)
        train_loader = torch.utils.data.DataLoader(fxc, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

        fxv = ADataset(images_filepaths=generateImgFilePaths(base_48_val, False), transform=trans)
        val_loader = torch.utils.data.DataLoader(fxv, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)
    # If Difficult
    elif datas == "diff":
        fxc = ADataset(images_filepaths=generateImgFilePaths(main_train, False), transform=trans)
        train_loader = torch.utils.data.DataLoader(fxc, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

        fxv = ADataset(images_filepaths=generateImgFilePaths(main_val, False), transform=trans)
        val_loader = torch.utils.data.DataLoader(fxv, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)
    # If Main
    else:
        fxc = ADataset(images_filepaths=generateImgFilePaths(additional_train, False), transform=trans)
        fxc1 = ADataset(images_filepaths=generateImgFilePaths(main_train, False), transform=trans)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([
                fxc,
                fxc1
            ]),
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        fxv = ADataset(images_filepaths=generateImgFilePaths(additional_val, False), transform=trans)
        fxv1 = ADataset(images_filepaths=generateImgFilePaths(main_val, False), transform=trans)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([
                fxv,
                fxv1
            ]),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not prev_classes:
        model = Net(num_classes)
    else:
        model = Net(prev_classes)
        if prev_name == "Benchmark model":
            state_dict = torch.load("benchmark.pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        else:
            state_dict = torch.load("generated/" + prev_name + ".pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(state_dict)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.fc2 = nn.Linear(300, num_classes)

    model.to(device)

    def validation(train_loss):
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            data = data.float()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            if torch.cuda.is_available():
                loss = F.nll_loss(output, target, size_average=False).cuda()
            else:
                loss = F.nll_loss(output, target, size_average=False)
            validation_loss += loss.data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        validation_loss /= len(val_loader.dataset)

        with open('stats.bin', 'rb+') as f:
            info = pickle.load(f)
            info["avg_train_loss"].append(float(train_loss))
            info["val_loss"].append(float(validation_loss))
            val_acc = float(100. * correct / len(val_loader.dataset))
            info["val_accuracy"] = val_acc
            f.seek(0)
            pickle.dump(info, f)

        return validation_loss, val_acc

    def train(epoch):
        model.train()
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.float()
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if torch.cuda.is_available():
                loss = F.nll_loss(output, target).cuda()
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            temp_loss = loss.data.item()
            running_loss += temp_loss*data.size(0)
            if batch_idx % log_interval == 0:
                with open('stats.bin', 'rb+') as f:
                    info = pickle.load(f)
                    info["train_epoch"] = int(epoch)
                    info["train_loss"] = float(temp_loss)
                    f.seek(0)
                    pickle.dump(info, f)
        running_loss /= len(train_loader.dataset)
        return running_loss

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_norm, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    temp = 10
    temp_epoch = None
    for epoch in range(1, epochs + 1):
        with open('stats.bin', 'rb') as f:
            info = pickle.load(f)
            if not info["running"]:
                print("Stopped!")
                break
        train_loss = train(epoch)
        val_metrics = validation(train_loss)
        val = val_metrics[0]
        val_acc = val_metrics[1]
        if epoch % step:
            scheduler.step()
        if val < temp:
            temp = val
            temp_epoch = epoch
            model_file = 'generated/' + name + '.pth'
            torch.save(model.state_dict(), model_file)
            with open('generated/' + name + '.json', "w") as f:
                f.write(json.dumps({
                    "method": "Train further",
                    "num_classes": num_classes,
                    "epochs": epoch,
                    "val_acc": val_acc,
                    "val_loss": val,
                    "train_loss": train_loss,
                    "lr": lr,
                    "momentum": momentum,
                    "gamma": decay,
                    "l2_norm": l2_norm
                }, indent=4))
            with open('stats.bin', 'rb+') as f:
                info = pickle.load(f)
                info["saved"] = True
                f.seek(0)
                pickle.dump(info, f)
        elif thres_epoch and temp_epoch and (epoch - temp_epoch > thres_epoch):
            with open('stats.bin', 'rb+') as f:
                info = pickle.load(f)
                info["running"] = False
                print("Threshold epoch reached!")
                f.seek(0)
                pickle.dump(info, f)
            break

        if thres_acc and val_acc > thres_acc:
            with open('stats.bin', 'rb+') as f:
                info = pickle.load(f)
                info["running"] = False
                print("Threshold accuracy reached!")
                f.seek(0)
                pickle.dump(info, f)
            break

    with open('stats.bin', 'rb+') as f:
        info = pickle.load(f)
        info["running"] = False
        print("Epochs completed!")
        f.seek(0)
        pickle.dump(info, f)


# For incremental learning
def inc_train_model(num_classes, batch_size, epochs, lr, decay, step, name, datas, prev_classes, prev_name, aug):
    torch.manual_seed(1)
    log_interval = 180

    if aug:
        trans = train_data_transforms
    else:
        trans = data_transforms

    # If GTSRB_48
    if datas == "base":
        fxc = ADataset(images_filepaths=generateImgFilePathsInc(base_48_train, prev_classes), transform=trans)
        train_loader = torch.utils.data.DataLoader(fxc, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

        fxv = ADataset(images_filepaths=generateImgFilePathsInc(base_48_val, prev_classes), transform=trans)
        val_loader = torch.utils.data.DataLoader(fxv, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)
    # If Difficult
    elif datas == "diff":
        fxc = ADataset(images_filepaths=generateImgFilePathsInc(main_train, prev_classes), transform=trans)
        train_loader = torch.utils.data.DataLoader(fxc, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

        fxv = ADataset(images_filepaths=generateImgFilePathsInc(main_val, prev_classes), transform=trans)
        val_loader = torch.utils.data.DataLoader(fxv, batch_size=batch_size, shuffle=False, num_workers=0,
                                                 pin_memory=True)
    # If Main
    else:
        fxc = ADataset(images_filepaths=generateImgFilePathsInc(additional_train, prev_classes), transform=trans)
        fxc1 = ADataset(images_filepaths=generateImgFilePathsInc(main_train, prev_classes), transform=trans)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([
                fxc,
                fxc1
            ]),
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )

        fxv = ADataset(images_filepaths=generateImgFilePathsInc(additional_val, prev_classes), transform=trans)
        fxv1 = ADataset(images_filepaths=generateImgFilePathsInc(main_val, prev_classes), transform=trans)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([
                fxv,
                fxv1
            ]),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(prev_classes)
    if prev_name == "Benchmark model":
        state_dict = torch.load("benchmark.pth",
                                map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    else:
        state_dict = torch.load("generated/" + prev_name + ".pth",
                                map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)

    extended = ExtendNet(model, num_classes-prev_classes)
    extended.to(device)
    freeze_params(extended.fc2, prev_classes)

    def extend_validation(train_loss):
        extended.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            data = data.float()
            data = data.to(device)
            target = target.to(device)
            output = extended(data)
            if torch.cuda.is_available():
                loss = F.nll_loss(output, target, size_average=False).cuda()
            else:
                loss = F.nll_loss(output, target, size_average=False)
            validation_loss += loss.data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        validation_loss /= len(val_loader.dataset)

        with open('stats.bin', 'rb+') as f:
            info = pickle.load(f)
            info["avg_train_loss"].append(float(train_loss))
            info["val_loss"].append(float(validation_loss))
            val_acc = float(100. * correct / len(val_loader.dataset))
            info["val_accuracy"] = val_acc
            f.seek(0)
            pickle.dump(info, f)

        return validation_loss, val_acc

    def extend_training(epoch, loss_fn):
        extended.train()
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.float()
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = extended(data)
            if torch.cuda.is_available():
                loss = loss_fn(output, target).cuda()
            else:
                loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            temp_loss = loss.data.item()
            running_loss += temp_loss * data.size(0)
            if batch_idx % log_interval == 0:
                with open('stats.bin', 'rb+') as f:
                    info = pickle.load(f)
                    info["train_epoch"] = int(epoch)
                    info["train_loss"] = float(temp_loss)
                    f.seek(0)
                    pickle.dump(info, f)
        running_loss /= len(train_loader.dataset)
        return running_loss

    extend_cost = extended_cost(0.8, 43, 5)
    optimizer = torch.optim.Adam(extended.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    temp = 10
    for epoch in range(1, epochs + 1):
        with open('stats.bin', 'rb') as f:
            info = pickle.load(f)
            if not info["running"]:
                print("Stopped!")
                break
        train_loss = extend_training(epoch, extend_cost)
        val_metrics = extend_validation(train_loss)
        val = val_metrics[0]
        val_acc = val_metrics[1]
        if epoch % step:
            scheduler.step()
        if val < temp:
            temp = val
            model_file = 'generated/' + name + '.pth'
            torch.save(extended.state_dict(), model_file)
            with open('generated/' + name + '.json', "w") as f:
                f.write(json.dumps({
                    "method": "Incremental Learning",
                    "num_classes": num_classes,
                    "epochs": epoch,
                    "val_acc": val_acc,
                    "val_loss": val,
                    "train_loss": train_loss,
                    "lr": lr,
                    "momentum": 0,
                    "gamma": decay,
                    "l2_norm": 0
                }, indent=4))
            with open('stats.bin', 'rb+') as f:
                info = pickle.load(f)
                info["saved"] = True
                f.seek(0)
                pickle.dump(info, f)

    with open('stats.bin', 'rb+') as f:
        info = pickle.load(f)
        info["running"] = False
        print("Epochs completed!")
        f.seek(0)
        pickle.dump(info, f)


# For model evaluation
def evaluate_model(name, nclasses, dataset, job_num):
    print('Evaluation started')
    reset_eval_stats(job_num)
    result = []
    try:
        with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
            info = pickle.load(t)
            info['eval_curr'] = 0
            t.seek(0)
            pickle.dump(info, t)

        if not name:
            state_dict = torch.load("benchmark.pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            num_classes = 43
        else:
            state_dict = torch.load("generated/" + name + ".pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            with open("generated/" + name + ".json", 'r') as f:
                info = json.load(f)
                num_classes = info["num_classes"]

        model = Net(num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        num = nclasses

        if dataset == "base" or dataset == "orig":
            loc = 'Base_48'
            fldr = base_48_test
        else:
            loc = 'Main'
            fldr = main_test

        with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
            info = pickle.load(t)
            info['eval_tot'] = 0
            t.seek(0)
            pickle.dump(info, t)
        eval_tot = 0

        if dataset == "main":
            loc = 'Additional'
            fldr = additional_test
            loc1 = 'Main'
            fldr1 = main_test

            for cls in range(num):
                for _ in os.listdir(fldr + str(cls)):
                    eval_tot += 1
                for _ in os.listdir(fldr1 + str(cls)):
                    eval_tot += 1

            with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
                info = pickle.load(t)
                info['eval_tot'] = eval_tot
                t.seek(0)
                pickle.dump(info, t)

            for cls in range(num):
                for f in os.listdir(fldr + str(cls)):
                    pred = inference(model, fldr + str(cls) + '/' + f)

                    # For progress stats
                    with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
                        info = pickle.load(t)
                        info['eval_curr'] += 1
                        t.seek(0)
                        pickle.dump(info, t)

                    trans = mongo.db.transformations.find_one({"name": f, "class": str(cls)})
                    if trans:
                        trans = trans["transformations"]
                    else:
                        trans = []
                    result.append({"Location": "static/" + loc + "/Test/" + str(cls) + '/' + f, "Actual": cls,
                                   "Pred": int(pred[1]), "Transformations": trans, "Conf": math.exp(float(pred[0]))})
                for f in os.listdir(fldr1 + str(cls)):
                    pred = inference(model, fldr1 + str(cls) + '/' + f)

                    # For progress stats
                    with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
                        info = pickle.load(t)
                        info['eval_curr'] += 1
                        t.seek(0)
                        pickle.dump(info, t)

                    trans = mongo.db.transformations.find_one({"name": f, "class": str(cls)})
                    if trans:
                        trans = trans["transformations"]
                    else:
                        trans = []
                    result.append({"Location": "static/" + loc1 + "/Test/" + str(cls) + '/' + f, "Actual": cls,
                                   "Pred": int(pred[1]), "Transformations": trans, "Conf": math.exp(float(pred[0]))})
        else:
            for cls in range(num):
                for _ in os.listdir(fldr + str(cls)):
                    eval_tot += 1

            with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
                info = pickle.load(t)
                info['eval_tot'] = eval_tot
                t.seek(0)
                pickle.dump(info, t)

            for cls in range(num):
                for f in os.listdir(fldr + str(cls)):
                    pred = inference(model, fldr + str(cls) + '/' + f)

                    # For progress stats
                    with open('eval_stats' + str(job_num) + '.bin', 'rb+') as t:
                        info = pickle.load(t)
                        info['eval_curr'] += 1
                        t.seek(0)
                        pickle.dump(info, t)

                    trans = mongo.db.transformations.find_one({"name": f, "class": str(cls)})
                    if trans:
                        trans = trans["transformations"]
                    else:
                        trans = []
                    result.append({"Location": "static/" + loc + "/Test/" + str(cls) + '/' + f, "Actual": cls,
                                   "Pred": int(pred[1]), "Transformations": trans, "Conf": math.exp(float(pred[0]))})

        return {"result": result}
    except Exception as e:
        print(e)
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_dataset', methods=['POST'])
def get_dataset():
    req_data = request.get_json()
    origInfo = {}

    if req_data["dataset"] == "GTSRB Dataset":
        fldr = "static/Base_48/"
        num = 43
    elif req_data["dataset"] == "GTSRB_48 Dataset":
        fldr = "static/Base_48/"
        num = 48
    elif req_data["dataset"] == "Main Dataset":
        fldr = "static/Additional/"
        fldr1 = "static/Main/"
        num = 48
    else:
        fldr = "static/Main/"
        num = 48

    if req_data["set"] == "Training Dataset":
        dataset = "Train/"
    elif req_data["set"] == "Test Dataset":
        dataset = "Test/"
    else:
        dataset = "Val/"

    for i in range(num):
        if req_data["dataset"] == "Main Dataset":
            origInfo[str(i)] = ["Additional/" + dataset + str(i) + "/" + s for s in os.listdir(fldr + dataset + str(i) + "/")]
            origInfo[str(i)].extend(["Main/" + dataset + str(i) + "/" + s for s in os.listdir(fldr1 + dataset + str(i) + "/")])
        else:
            origInfo[str(i)] = os.listdir(fldr + dataset + str(i) + "/")
    return origInfo


@app.route('/addImages/Train')
def Train():
    addInfo = {}
    num = max([int(i) for i in os.listdir(additional_train)])
    for i in range(int(num) + 1):
        files = os.listdir(additional_train + str(i) + "/")
        files.sort(key=lambda x: os.path.getmtime(additional_train + str(i) + "/" + x), reverse=True)
        addInfo[str(i)] = files
    return addInfo


@app.route('/addImages/Test')
def Test():
    addInfo = {}
    num = max([int(i) for i in os.listdir(additional_test)])
    for i in range(int(num) + 1):
        files = os.listdir(additional_test + str(i) + "/")
        files.sort(key=lambda x: os.path.getmtime(additional_test + str(i) + "/" + x), reverse=True)
        addInfo[str(i)] = files
    return addInfo


@app.route('/addImages/Val')
def Val():
    addInfo = {}
    num = max([int(i) for i in os.listdir(additional_val)])
    for i in range(int(num) + 1):
        files = os.listdir(additional_val + str(i) + "/")
        files.sort(key=lambda x: os.path.getmtime(additional_val + str(i) + "/" + x), reverse=True)
        addInfo[str(i)] = files
    return addInfo


@app.route('/addClass', methods=['POST'])
def addClass():
    req_data = request.get_json()
    try:
        os.mkdir(additional_train + str(req_data['class']))
        os.mkdir(additional_test + str(req_data['class']))
        os.mkdir(additional_val + str(req_data['class']))
        return "Successfully added new class"
    except Exception as e:
        print(e)
        return "Invalid request", 400

@app.route('/removeClass', methods=['POST'])
def removeClass():
    req_data = request.get_json()
    try:
        shutil.rmtree(additional_train + str(req_data['class']))
        shutil.rmtree(additional_test + str(req_data['class']))
        shutil.rmtree(additional_val + str(req_data['class']))
        return "Successfully removed last class"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/cleartemp')
def cleartemp():
    shutil.rmtree(temp)
    os.mkdir('temp')
    return "Temp folder cleared"


@app.route('/addImages', methods=['POST'])
def addImages():
    if '_method' in request.headers:
        req_data = request.get_json()
        try:
            shutil.rmtree(os.path.join(temp, req_data['uid']))
            return "Delete Successful"
        except Exception as e:
            print(e)
            return "Invalid request", 400
    else:
        if 'images' not in request.files:
            print('No file part')
            return "No File Part", 400
        file = request.files["images"]
        if file.filename == '':
            print('No selected file')
            return "No Selected File", 400
        id = str(uuid.uuid1())
        os.mkdir(os.path.join(temp, id))
        file.save(os.path.join(temp, id, file.filename))
        return id, 201


@app.route('/finalAddImages', methods=['POST'])
def finalAddImages():
    req_data = request.get_json()
    img = None
    try:
        if req_data['num']:
            transformer = transform(req_data['options'])
        else:
            transformer = None
        scan_dir(temp, req_data['num'], transformer, req_data['class'][6:], mongo, additional_temp)
        if req_data['add_test']:
            train, val, test = random_segregate_all(req_data['ratio'], additional_temp)
            for y in train:
                shutil.move(os.path.join(additional_temp, y), os.path.join(additional_train, req_data['class'][6:], y))
            for y in val:
                shutil.move(os.path.join(additional_temp, y), os.path.join(additional_val, req_data['class'][6:], y))
            for y in test:
                shutil.move(os.path.join(additional_temp, y), os.path.join(additional_test, req_data['class'][6:], y))
        else:
            if req_data['seg'] == 'Smart Segregation':
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = Net()
                state_dict = torch.load("benchmark.pth",
                                        map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                feature_model = FNet(model)
                feature_model.to(device)
                train, val, img = smart_segregate(feature_model, device, 2, req_data['ratio'], additional_temp, pil_loader,
                                             data_transforms)
            else:
                train, val = random_segregate(req_data['ratio'], additional_temp)

            for y in train:
                shutil.move(os.path.join(additional_temp, y), os.path.join(additional_train, req_data['class'][6:], y))
            for y in val:
                shutil.move(os.path.join(additional_temp, y), os.path.join(additional_val, req_data['class'][6:], y))

        shutil.rmtree(temp)
        os.mkdir(temp)
        if req_data['seg'] == 'Smart Segregation':
            return img
        else:
            return "Successful"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/delImages', methods=['POST'])
def delImages():
    req_data = request.get_json()
    try:
        cls = req_data['file'].split("/")[6]
        name = req_data['file'].split("/")[7]
        mongo.db.transformations.delete_one({"name": name, "class": cls})
        filepath = req_data['file'].split("/")[6:]
        filepath = "/".join(filepath)
        if req_data['type'] == "Training Dataset":
            os.remove(additional_train + filepath)
        elif req_data['type'] == "Test Dataset":
            os.remove(additional_test + filepath)
        else:
            os.remove(additional_val + filepath)
        return "Successfully deleted"
    except Exception as e:
        print(e)
        return "Invalid request"


@app.route('/delmulti', methods=['POST'])
def delmulti():
    req_data = request.get_json()
    try:
        for x in req_data["files"]:
            cls = x.split("/")[6]
            name = x.split("/")[7]
            mongo.db.transformations.delete_one({"name": name, "class": cls})
            filepath = x.split("/")[6:]
            filepath = "/".join(filepath)
            if req_data['type'] == "Training Dataset":
                os.remove(additional_train + filepath)
            elif req_data['type'] == "Test Dataset":
                os.remove(additional_test + filepath)
            else:
                os.remove(additional_val + filepath)
        return "Successfully deleted images"
    except Exception as e:
        print(e)
        return "Invalid request"


@app.route('/moveImages', methods=['POST'])
def moveImages():
    req_data = request.get_json()
    try:
        filepath = req_data['file'].split("/")[6:]
        filepath = "/".join(filepath)
        if req_data['type'] == "Training Dataset":
            fromdir = additional_train + filepath
        elif req_data['type'] == "Test Dataset":
            fromdir = additional_test + filepath
        else:
            fromdir = additional_val + filepath
        if req_data['toType'] == "Train":
            todir = additional_train + filepath
        elif req_data['toType'] == "Test":
            todir = additional_test + filepath
        else:
            todir = additional_val + filepath
        shutil.move(fromdir, todir)
        return "Successfully moved"
    except Exception as e:
        print(e)
        return "Invalid request"


@app.route('/movemulti', methods=['POST'])
def movemulti():
    req_data = request.get_json()
    try:
        for x in req_data["files"]:
            filepath = x.split("/")[6:]
            filepath = "/".join(filepath)
            if req_data['type'] == "Training Dataset":
                fromdir = additional_train + filepath
            elif req_data['type'] == "Test Dataset":
                fromdir = additional_test + filepath
            else:
                fromdir = additional_val + filepath
            if req_data['toType'] == "Train":
                todir = additional_train + filepath
            elif req_data['toType'] == "Test":
                todir = additional_test + filepath
            else:
                todir = additional_val + filepath
            shutil.move(fromdir, todir)
        return "Successfully moved images"
    except Exception as e:
        print(e)
        return "Invalid request"


@app.route('/saveCopy', methods=['POST'])
def saveCopy():
    req_data = request.get_json()
    try:
        image = readb64(req_data['image'])
        filepath = req_data['path'].split('/')[6:]
        name = filepath[-1]
        name = name.split('.')
        name[-2] = name[-2] + "_copy"
        name = ".".join(name)
        filepath[-1] = name
        filepath = "/".join(filepath)
        if req_data['type'] == "Training Dataset":
            cv2.imwrite(os.path.join(additional_train, filepath), image)
        elif req_data['type'] == "Test Dataset":
            cv2.imwrite(os.path.join(additional_test, filepath), image)
        else:
            cv2.imwrite(os.path.join(additional_val, filepath), image)
        return "Successful"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/save', methods=['POST'])
def save():
    req_data = request.get_json()
    try:
        image = readb64(req_data['image'])
        filepath = req_data['path'].split("/")[6:]
        filepath = "/".join(filepath)
        if req_data['type'] == "Training Dataset":
            cv2.imwrite(os.path.join(additional_train, filepath), image)
        elif req_data['type'] == "Test Dataset":
            cv2.imwrite(os.path.join(additional_test, filepath), image)
        else:
            cv2.imwrite(os.path.join(additional_val, filepath), image)
        return "Successful"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/evaluate', methods=['POST'])
def evaluate():
    req_data = request.get_json()
    try:
        job = q.enqueue(evaluate_model, req_data['name'], req_data['nclasses'], req_data['dataset'],
                        req_data['job_num'], job_timeout=3600)
        global job_id
        job_id[req_data['job_num']] = job
        return "Evaluation started successfully"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/eval_result', methods=['POST'])
def eval_result():
    req_data = request.get_json()
    if job_id[req_data['job_num']]:
        return job_id[req_data['job_num']].result
    else:
        return "Not available"


@app.route('/heatmap', methods=['POST'])
def heatmap():
    req_data = request.get_json()
    try:
        if not req_data["name"]:
            state_dict = torch.load("benchmark.pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            num_classes = 43
        else:
            state_dict = torch.load("generated/" + req_data["name"] + ".pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            with open("generated/" + req_data["name"] + ".json", 'r') as f:
                info = json.load(f)
                num_classes = info["num_classes"]

        model = Net(num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        image = np.array(pil_loader(req_data['loc']))

        if req_data['cap']:
            blended, normal = generate_heat_map(model, image, data_transforms)
            return {"blended": blended, "normal": normal}
        else:
            string = anchor_heat_map(model, num_classes, image, data_transforms)
            return string

    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/evaluateImages', methods=['POST'])
def evaluateImages():
    req_data = request.get_json()
    try:
        result = []
        if not req_data["name"]:
            state_dict = torch.load("benchmark.pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            num_classes = 43
        else:
            state_dict = torch.load("generated/" + req_data["name"] + ".pth",
                                    map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            with open("generated/" + req_data["name"] + ".json", 'r') as f:
                info = json.load(f)
                num_classes = info["num_classes"]

        model = Net(num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        for dirs in os.listdir(temp):
            f = os.listdir(temp + dirs)[0]
            pred = inference(model, temp + dirs + "/" + f)
            img = writeb64(temp + dirs + "/" + f)
            result.append({"Image": img, "Pred": int(pred[1]), "Confidence": math.exp(float(pred[0]))})
        shutil.rmtree(temp)
        os.mkdir(temp)
        return {"result": result}

    except Exception as e:
        print(e)
        return "Invalid request", 400


# For visualizing classes on tnse
@app.route('/visualize', methods=['POST'])
def visualize():
    req_data = request.get_json()
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not req_data["name"]:
            state_dict = torch.load("benchmark.pth", map_location=device)
            num_classes = 43
        else:
            state_dict = torch.load("generated/" + req_data["name"] + ".pth", map_location=device)
            with open("generated/" + req_data["name"] + ".json", 'r') as f:
                info = json.load(f)
                num_classes = info["num_classes"]

        model = Net(num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        X_features = []
        labels = []

        for dirs in os.listdir(temp):
            f = os.listdir(temp + dirs)[0]
            data = data_transforms(image=np.array(pil_loader(temp + dirs + "/" + f)))["image"]
            data = data.view(1, data.shape[0], data.shape[1], data.shape[2])
            data = Variable(data)
            data = data.float()
            data = data.to(device)
            features = model(data)
            pred = features.data.max(1, keepdim=True)
            labels.append(int(pred[1]))
            X_features.append(features.cpu().detach().numpy().squeeze())

        X_features = np.array(X_features)
        labels = np.array(labels)
        img = tsne_plot(X_features, labels, 2)

        shutil.rmtree(temp)
        os.mkdir(temp)
        return img
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/trainmodel', methods=['POST'])
def trainmodel():
    req_data = request.get_json()
    try:
        reset_stats()
        with open('stats.bin', 'rb+') as f:
            info = pickle.load(f)
            info["running"] = True
            f.seek(0)
            pickle.dump(info, f)
        if not req_data["inc"]:
            job = q.enqueue(train_model, req_data['num_classes'], req_data['batch_size'], req_data['epochs'],
                            req_data['lr'], req_data['momentum'],
                            req_data['decay'], req_data['step'], req_data['l2_norm'], req_data['name'],
                            req_data["thres_acc"], req_data["thres_epoch"], req_data["data"],
                            req_data["prev_classes"], req_data["prev_name"], req_data["freeze"],
                            req_data['aug'], job_timeout=3600)
        else:
            job = q.enqueue(inc_train_model, req_data['num_classes'], req_data['batch_size'], req_data['epochs'],
                            req_data['lr'], req_data['decay'], req_data['step'],req_data['name'],
                            req_data["data"], req_data["prev_classes"], req_data["prev_name"],
                            req_data['aug'], job_timeout=3600)
        return "Training Started Successful"
    except Exception as e:
        print(e)
        return "Invalid request", 400


@app.route('/stoptraining')
def stoptraining():
    try:
        q.empty()
        with open('stats.bin', 'rb+') as f:
            info = pickle.load(f)
            info["running"] = False
            f.seek(0)
            pickle.dump(info, f)
        return "Training successfully stopped"
    except Exception as e:
        print(e)
        return "Error", 500


@app.route('/traininfo')
def traininfo():
    with open('stats.bin', 'rb') as f:
        info = pickle.load(f)
        return info


@app.route('/modelinfo')
def modelinfo():
    info = os.listdir("generated")
    info = sorted([x for x in info if x[-3:] == "pth"])
    return {"result": info}


@app.route('/modelstats', methods=['POST'])
def modelstats():
    req_data = request.get_json()
    try:
        if req_data["name"] == "Benchmark Model":
            info = {
                "method": "Train further",
                "num_classes": 43,
                "epochs": 193,
                "val_acc": 97.74740572007087,
                "val_loss": 0.17600008994340897,
                "train_loss": 1.2196484511804764,
                "l2_norm": 0.00001,
                "lr": 0.007,
                "momentum": 0.8,
                "gamma": 0.9
            }
            return info
        else:
            with open("generated/" + req_data["name"] + ".json", 'r') as f:
                info = json.load(f)
                return info
    except Exception as e:
        print(e)
        return "Error", 500


@app.route('/eval_stats_clear')
def eval_stats_clear():
    try:
        for i in range(5):
            with open('eval_stats' + str(i) + '.bin', 'wb') as f:
                pickle.dump(eval_stats, f)
        return "Successful"
    except Exception as e:
        print(e)
        return "Failed"


@app.route('/evalprogress')
def evalprogress():
    progs = []
    for i in range(5):
        with open('eval_stats' + str(i) + '.bin', 'rb') as f:
            info = pickle.load(f)
        prog = 0
        if info['eval_tot'] != 0:
            prog = info['eval_curr']/info['eval_tot'] * 100.0
        progs.append(int(prog))
    return {"prog": progs}


if __name__ == "__main__":
    app.run(host='0.0.0.0')
