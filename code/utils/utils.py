import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomResizedCrop, RandomRotation, RandomVerticalFlip, Resize
from utils.dataset import CIFAR10_truncated, ImageFolder_custom, CIFAR10_truncated_constrastive
import pandas as pd 
import random 
import math 

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency=1., consistency_rampup=40):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir + './train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir + './val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array(
        [int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_isic_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    isic_train_ds = ImageFolder_custom(datadir + './train/', transform=transform)
    isic_test_ds = ImageFolder_custom(datadir + './valid/', transform=transform)

    X_train, y_train = np.array([s[0] for s in isic_train_ds.samples]), np.array(
        [int(s[1]) for s in isic_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in isic_test_ds.samples]), np.array([int(s[1]) for s in isic_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_dernmet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir + './train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir + './test/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array(
        [int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10' or dataset == 'cifar10-con':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'isic':
        X_train, y_train, X_test, y_test = load_isic_data(datadir)
    elif dataset == 'dermnet':
        X_train, y_train, X_test, y_test = load_dernmet_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'isic':
            K = 4
        elif dataset == 'dermnet':
            K = 23
        else:
            K = 10
        if dataset in ['tinyimagenet', 'dermnet']:
            times = [0 for i in range(K)]
            contain = []
            if K % num != 0:
                split = np.random.choice([*list(range(K)), *np.random.choice(range(K), math.ceil(K / num) * num - K, replace=False)], size=(math.ceil(K/num), num), replace=False)
            else:
                split = np.random.choice(range(K), size=(K//num, num), replace=False)
            for i in range(n_parties):
                if i < math.ceil(K / num):
                    current = list(split[i])
                else:
                    current = np.random.choice(range(K), num)
                for j in current:
                    times[j] += 1
                contain.append(current)
            net_dataidx_map = {i:np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1
            key = list(net_dataidx_map.keys())
            random.shuffle(key)
            random_net_dataidx_map = {}
            for i, k in enumerate(key):
                random_net_dataidx_map[i] = net_dataidx_map[k]
            net_dataidx_map = random_net_dataidx_map
        elif num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            #times=[0 for i in range(10)]
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    #samples = np.asarray([2723, 3000, 2019, 531, 1643, 138, 160, 385])
                    #ind = np.random.choice(range(0, K), 1, p=samples/np.sum(samples))[0]
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        #net_dataidx_map[j]=np.append(net_dataidx_map[j],np.random.choice(split[ids], min(2000, len(split[ids])*4)))
                        ids+=1

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    # print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X

def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel

def tsne_visualize(model, dataloader, device, save_file='tsne.png'):
    classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'train']
    print('TSNE VISUALIZE !!!')

    def _scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            _, feature, _ = model(x)
            features.append(feature.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = _scale_to_01_range(tx)
    ty = _scale_to_01_range(ty)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label in range(10):
        indices = [i for i, l in enumerate(labels) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, label=classes[label])
    ax.legend(loc='best')
    plt.savefig(save_file)
    plt.close()

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    # print("x:",x)
                    # print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                # print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _, _, out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss


def save_model(model, model_index, args):
    print("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

from math import sqrt 
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,min(27, row*size+i),min(27, col*size+j)] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0):
    if dataset in ('cifar10', 'cifar10-con'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        elif dataset == 'cifar10-con':
            dl_obj = CIFAR10_truncated_constrastive
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + './train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir + './val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'isic':
        dl_obj = ImageFolder_custom
        input_size = 112
        transform_train = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + './train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir + './valid/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'dermnet':
        dl_obj = ImageFolder_custom
        input_size = 224
        transform_train = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + './train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir + './test/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds
