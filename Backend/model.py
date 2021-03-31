import torch
import torch.nn as nn
import torch.nn.functional as F


# Main MicronNet model
class Net(nn.Module):
    def __init__(self, nclasses = 43):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, nclasses)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3( self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4( self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)


# For feature extraction
class FNet(nn.Module):
    def __init__(self, model):
        super(FNet, self).__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.maxpool2 = model.maxpool2
        self.conv3 = model.conv3
        self.maxpool3 = model.maxpool3
        self.conv4 = model.conv4
        self.maxpool4 = model.maxpool4
        self.conv2_drop = model.conv2_drop
        self.conv3_drop = model.conv3_drop
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.conv0_bn = model.conv0_bn
        self.conv1_bn = model.conv1_bn
        self.conv2_bn = model.conv2_bn
        self.conv3_bn = model.conv3_bn
        self.conv4_bn = model.conv4_bn
        self.dense1_bn = model.dense1_bn
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3( self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4( self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = torch.tanh(self.fc1(x))
        return x


# For incremental learning
class ExtendNet(nn.Module):
    def __init__(self, model, newclasses):
        super(ExtendNet, self).__init__()
        self.og_classes = model.fc2.weight.shape[0]
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.maxpool2 = model.maxpool2
        self.conv3 = model.conv3
        self.maxpool3 = model.maxpool3
        self.conv4 = model.conv4
        self.maxpool4 = model.maxpool4
        self.conv2_drop = model.conv2_drop
        self.conv3_drop = model.conv3_drop
        self.fc1 = model.fc1
        self.fc2 = nn.Linear(300, self.og_classes + newclasses)
        with torch.no_grad():
            self.fc2.weight[:self.og_classes, :] = model.fc2.weight.data
        self.conv0_bn = model.conv0_bn
        self.conv1_bn = model.conv1_bn
        self.conv2_bn = model.conv2_bn
        self.conv3_bn = model.conv3_bn
        self.conv4_bn = model.conv4_bn
        self.dense1_bn = model.dense1_bn

        for child in self.children():
            for param in child.parameters():
                param.requires_grad = False

        self.fc2.weight.requires_grad = True
        self.fc2.bias.requires_grad = True

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)


def freeze_params(layer, og_classes):
    weight_multiplier = torch.ones(layer.weight.shape).to(layer.weight.device)
    weight_multiplier[:og_classes, :] = 0
    bias_multiplier = torch.ones(layer.weight.shape[0]).to(layer.bias.device)
    bias_multiplier[:og_classes] = 0

    def freezing_weight(grad, weight_multiplier):
        newgrad = grad * weight_multiplier
        return newgrad

    def freezing_bias(grad, bias_multiplier):
        return grad * bias_multiplier

    freezing_hook_weight = lambda grad: freezing_weight(grad, weight_multiplier)
    freezing_hook_bias = lambda grad: freezing_bias(grad, bias_multiplier)

    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)
    bias_hook_handle = layer.bias.register_hook(freezing_hook_bias)
    return


class extended_cost(nn.Module):
  def __init__(self, c, A, B):
    super(extended_cost, self).__init__()
    self.eps = (c * A)/float(A+B)
    self.og_classes = A
    self.new_classes = B

  def forward(self, outputs, labels):
    bsize = labels.shape[0]
    labels = labels.view(-1)
    # print(outputs.shape)
    outputs1 = outputs[range(outputs.shape[0]), labels]
    outputs2 = outputs[range(outputs.shape[0]), :self.og_classes]
    # print(outputs2.shape)
    loss = -torch.sum((1-self.eps)*outputs1 + (self.eps/(1.0*self.og_classes))*torch.sum(torch.exp(outputs2)*outputs2, dim=1))
    return loss/bsize
