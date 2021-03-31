from torch.autograd import Variable
from random import sample
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import numpy as np

# For random segregation
def random_segregate(ratio, fldr):
    listOfFiles = os.listdir(fldr)
    lenVal = int((1-ratio) * len(listOfFiles))
    val = sample(listOfFiles, lenVal)
    train = [item for item in listOfFiles if item not in val]
    return train, val


def random_segregate_all(ratio, fldr):
    listOfFiles = os.listdir(fldr)
    lenTest = int((max(ratio[0], ratio[1]) - min(ratio[0], ratio[1])) / 100 * len(listOfFiles))
    lenVal = int((100 - max(ratio[0], ratio[1])) / 100 * len(listOfFiles))
    test = sample(listOfFiles, lenTest)
    x = [item for item in listOfFiles if item not in test]
    val = sample(x, lenVal)
    train = [item for item in x if item not in val]
    return train, val, test


# For plotting clusters
def tsne_plot(X_features, labels, n_jobs):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500, n_jobs=n_jobs, random_state=1)
    tsne_res = tsne.fit_transform(X_features)
    tsne_dict = {}
    tsne_dict["tsne-one"] = tsne_res[:,0]
    tsne_dict["tsne-two"] = tsne_res[:,1]
    tsne_dict["y"] = labels.squeeze()
    num_palettes = (np.unique(labels)).shape[0]

    plt.figure(figsize=(8,5))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="y",
        palette=sns.color_palette("husl", num_palettes),
        data=tsne_dict,
        legend=False,
        alpha=1
    )
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    return base64.b64encode(pic_IObytes.read()).decode()


# For smart segregation
def smart_segregate(model, device, n_jobs, ratio, fldr, pil_loader, data_transforms):
    X_train = []
    X_val = []
    model.eval()
    X_features = []
    X_images = []
    files = sorted(os.listdir(fldr))
    for fname in files:
        data = data_transforms(image=np.array(pil_loader(fldr + '/' + fname)))["image"]
        data = data.view(1, data.shape[0], data.shape[1], data.shape[2])
        data = Variable(data)
        data = data.float()
        data = data.to(device)
        features = model(data)
        X_features.append(features.cpu().detach().numpy().squeeze())
        X_images.append(fname)

    X_features = np.array(X_features)
    X_images = np.array(X_images)
    bandwidth = estimate_bandwidth(X_features, n_jobs=n_jobs, quantile=0.2, random_state=1)
    cluster = MeanShift(bandwidth, n_jobs=n_jobs).fit(X_features)

    img = tsne_plot(X_features, cluster.labels_, n_jobs)

    for i in range(cluster.labels_.max() + 1):
        temp = X_images[cluster.labels_.squeeze() == i]
        n_imgs = temp.shape[0]
        X_train += [temp[i] for i in range(0, int(n_imgs * ratio))]
        X_val += [temp[i] for i in range(int(n_imgs * ratio), n_imgs)]
    return X_train, X_val, img
