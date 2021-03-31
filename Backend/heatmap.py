import torch
from torch.autograd import Variable
import albumentations as A
from alibi.explainers import AnchorImage
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import cv2
import io
import base64
import numpy as np


# Heat map using captum
def generate_heat_map(model, image, data_transforms):
    data = data_transforms(image=image)["image"]
    data = data.float()
    data = data.view(1, data.size(0), data.size(1), data.size(2))
    data = Variable(data)
    x_prob = np.squeeze(model(data).detach().numpy())
    target = x_prob.argmax(axis=0)

    img_res = A.Resize(48, 48)(image=image)["image"]
    temp = np.transpose(img_res)
    sh = np.shape(temp)
    temp = np.reshape(temp, (1, sh[0], sh[1], sh[2]))
    img_tens = torch.from_numpy(temp)
    img_tens = img_tens.float()

    ig = IntegratedGradients(model)

    attr_ig, delta = ig.attribute(img_tens, baselines=img_tens * 0, return_convergence_delta=True,
                                  target=torch.tensor(target))
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))

    t = np.clip(attr_ig, a_min=0.0, a_max=None)
    min_vals = np.amin(t, axis=(0, 1, 2), keepdims=True)
    max_vals = np.amax(t, axis=(0, 1, 2), keepdims=True)

    normalized = (t - min_vals) / (max_vals - min_vals)
    clipped = np.clip(normalized, a_min=np.min(normalized), a_max=np.percentile(normalized, 99.9))

    blended, _ = viz.visualize_image_attr(attr_ig, img_res, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title="Overlayed Integrated Gradients", use_pyplot=False)

    normal, _ = viz.visualize_image_attr(None, clipped,
                                 method="original_image", title="Normalized attributes", use_pyplot=False)

    pic_IObytes = io.BytesIO()
    blended.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    blended = base64.b64encode(pic_IObytes.read()).decode()
    pic_IObytes.seek(0)
    normal.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    normal = base64.b64encode(pic_IObytes.read()).decode()

    return blended, normal


# Anchor heatmap
def anchor_heat_map(model, num_classes, image, data_transforms):

    def micronnet_prob(x):
        l = np.shape(x)[0]
        x_prob = np.zeros((l, num_classes))
        for i in range(l):
            data = data_transforms(image=x[i])["image"]
            data = data.float()
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            data = Variable(data)
            x_prob[i] = np.squeeze(model(data).detach().numpy())
        return x_prob

    image_shape = np.shape(image)
    explainer_anch = AnchorImage(micronnet_prob, image_shape, segmentation_fn='slic',
                                 segmentation_kwargs={'n_segments': 10, 'compactness': 20, 'sigma': .5},
                                 images_background=None)
    explanation_anch = explainer_anch.explain(image, threshold=.95, p_sample=.5, tau=0.15)
    img = explanation_anch.anchor
    img = img[:, :, ::-1]
    string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    return string
