import torch.nn as nn
import torch.nn.functional as F
from models.model import ResNet18_cifar10, ResNet50_cifar10, MLP_header, SimpleCNN_header, SimpleCNNMNIST_header, TinyVGG, GNResNet, SimpleCNNHAM10000_header
from models.resnet import GNResNet18, GNResNet50, GNResNet18RW
from torchvision import models
import timm 
from torchvision.models import shufflenet_v2_x0_5

class MyAdaptivePool2d(nn.Module):
    def forward(self, x):
        x = x[-1]
        x = F.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)
        return x 

class ModelFedCon(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        if base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel" or base_model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-ham10000':
            self.features = SimpleCNNHAM10000_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'resnet18':
            self.features = timm.create_model('resnet18', pretrained=False, num_classes=128)
            num_ftrs = 128
        elif base_model == 'shufflenet':
            self.features = nn.Sequential(*list(shufflenet_v2_x0_5().children())[:-2], MyAdaptivePool2d())
            num_ftrs = 192
        elif base_model == 'vgg16':
            self.features = nn.Sequential(timm.create_model('vgg16', features_only=True, pretrained=False), MyAdaptivePool2d())
            num_ftrs = 512
        elif base_model == 'tinyvgg':
            self.features = TinyVGG(input_dim=64, hidden_dims=[120, 84])
            num_ftrs = 84
        elif base_model == 'gnresnet':
            self.features = GNResNet(128)
            num_ftrs = 128
        elif base_model == 'gnresnet18':
            self.features = GNResNet18(num_classes=256)
            num_ftrs = 256 
        elif base_model == 'gnresnet50':
            self.features = GNResNet50(num_classes=256)
            num_ftrs = 256
        elif base_model == 'gnresnet18rw':
            self.features = GNResNet18RW(num_classes=256)
            num_ftrs = 256 

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        y = self.l3(x)
        return h, x, y


class ModelFedCon_noheader(nn.Module):

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()

        if base_model == "resnet50":
            basemodel = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18":
            basemodel = models.resnet18(pretrained=False)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet50-cifar10" or base_model == "resnet50-cifar100" or base_model == "resnet50-smallkernel":
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif base_model == "mlp":
            self.features = MLP_header()
            num_ftrs = 512
        elif base_model == 'simple-cnn':
            self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84
        elif base_model == 'simple-cnn-mnist':
            self.features = SimpleCNNMNIST_header(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)
            num_ftrs = 84

        #summary(self.features.to('cuda:0'), (3,32,32))
        #print("features:", self.features)
        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        p, h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        #print("h after:", h)
        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)

        y = self.l3(h)
        return p, h, y
