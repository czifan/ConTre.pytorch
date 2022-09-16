from copy import deepcopy
from numpy.matrixlib.defmatrix import matrix
import torch
import os
from utils import *
from models import *
import numpy as np
import logging
import random
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import torch.nn.functional as F 

class FedAvg(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = os.path.join(args.output_dir, args.trainer, args.dataset, args.partition, str(args.init_seed))
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = self._build_logging(os.path.join(self.output_dir, 'log.log'))
        self.printer = self.logger.info

        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta)
        self.net_dataidx_map = net_dataidx_map
        self.traindata_cls_counts = traindata_cls_counts

        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list = [i for i in range(args.n_parties)]
        party_list_rounds = []
        if n_party_per_round != args.n_parties:
            for i in range(args.rounds):
                party_list_rounds.append(random.sample(party_list, n_party_per_round))
        else:
            for i in range(args.rounds):
                party_list_rounds.append(party_list)
        self.party_list_rounds = party_list_rounds

        self.n_classes = len(np.unique(y_train))

        self.train_dl_gobal, self.test_dl, self.train_ds_global, self.test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

        self.nets, self.local_model_meta_data, self.layer_type = init_nets(args.net_config, args.n_parties, args)

        self.global_models, self.global_model_meta_data, self.global_layer_type = init_nets(args.net_config, 1, args)
        self.global_model = self.global_models[0]

        if args.server_momentum:
            self.moment_v = deepcopy(self.global_model.state_dict())
            for key in self.moment_v:
                self.moment_v[key] = 0

    def _build_logging(self, filename):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=filename,
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logger = logging.getLogger('')
        logger.addHandler(console)
        return logger

    def _compute_accuracy(self, model, dataloader, get_confusion_matrix=False):
        model.to(self.device)
        model.eval()

        true_labels, pred_labels = [], []

        loss = 0.0
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader):
                image = image.float().to(self.device)
                label = label.long().to(self.device)
                _, _, out = model(image)
                loss += criterion(out, label).item()
                true_labels.append(label.detach().cpu().numpy())
                pred_labels.append(out.argmax(dim=1).detach().cpu().numpy())
        true_labels = np.concatenate(true_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)
        accuracy = (true_labels == pred_labels).sum() * 1.0 / true_labels.shape[0]
        if get_confusion_matrix:
            return accuracy, None, loss / len(dataloader)
        return accuracy, loss / len(dataloader)

    def _local_train_net(self, round, nets_this_round):
        for net_id, net in nets_this_round.items():
            dataidxs = self.net_dataidx_map[net_id]
            self.printer(f'Training network {net_id}. n_training {len(dataidxs)}')
            train_dl_local, test_dl_local, _, _ = get_dataloader(self.args.dataset,
                                                                 self.args.datadir,
                                                                 self.args.batch_size,
                                                                 32,
                                                                 dataidxs, net_id=net_id, total=self.args.n_parties-1)
            net.to(self.device)
            net.train()
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)

            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            for epoch in range(self.args.epochs):
                epoch_loss = []
                for batch_idx, (image, label) in enumerate(train_dl_local):
                    image = image.float().to(self.device)
                    label = label.long().to(self.device)

                    optimizer.zero_grad()
                    image.requires_grad = True
                    label.requires_grad = False
                    _, _, out = net(image)
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                self.printer(f'Round: {round}\tEpoch: {epoch}\tLoss: {np.mean(epoch_loss):.6f}')

            test_acc, conf_matrix, _ = self._compute_accuracy(net, test_dl_local, get_confusion_matrix=True)
            self.printer(f'>> Test Accuracy: {test_acc}')
            net.to('cpu')
            self.printer(' ** Training complete **')

    def _global_train_net(self, round, nets_this_round):
        global_w = self.global_model.state_dict()
        if self.args.server_momentum:
            old_w = deepcopy(self.global_model.state_dict())

        total_data_points = sum([len(self.net_dataidx_map[r]) for r in range(self.args.n_parties)])
        fed_avg_freqs = [len(self.net_dataidx_map[r]) / total_data_points for r in range(self.args.n_parties)]

        for net_id, net in enumerate(nets_this_round.values()):
            net_param = net.state_dict()
            if net_id == 0:
                for key in net_param:
                    global_w[key] = net_param[key] * fed_avg_freqs[net_id]
            else:
                for key in net_param:
                    global_w[key] += net_param[key] * fed_avg_freqs[net_id]

        if self.args.server_momentum:
            delta_w = deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = old_w[key] - global_w[key]
                self.moment_v[key] = self.args.server_momentum * self.moment_v[key] + (1-self.args.server_momentum) * delta_w[key]
                global_w[key] = old_w[key] - self.moment_v[key]

        self.global_model.load_state_dict(global_w)

        self.printer(f'global n_test: {len(self.test_dl)}')
        self.global_model.to(self.device)
        test_acc, conf_matrix, _ = self._compute_accuracy(self.global_model, self.test_dl, get_confusion_matrix=True)
        self.printer('>> Round: %d\tGlobal Model Test accuracy: %f' % (round, test_acc))
        self.global_model.to('cpu')

        return test_acc

    def _report_relation(self, weight):
        for i in range(weight.shape[0]):
            context = ''
            for j in range(weight.shape[0]):
                sim = F.cosine_similarity(weight[i], weight[j], dim=0).item()
                context += f'{np.round(sim, 2)}\t'
            self.printer(context)

    def _before_train(self):
        return

    def _tsne_visualize(self, model, dataloader, save_file='tsne.png'):
        classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'train']
        print('TSNE VISUALIZE !!!')
        def _scale_to_01_range(x):
            value_range = (np.max(x) - np.min(x))
            starts_from_zero = x - np.min(x)
            return starts_from_zero / value_range

        model.to(self.device)
        model.eval()
        features, labels = [], []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(self.device), target.to(dtype=torch.int64).to(self.device)
                _, feature, _ = model(x)
                features.append(feature.detach().cpu().numpy())
                labels.append(target.detach().cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.savez(save_file.replace('.png', '.npz'), features=features, labels=labels)

    def run(self):
        best_acc = 0.0
        f = open(os.path.join(self.output_dir, 'test_acc.txt'), 'w')
        f.close()
        f = open(os.path.join(self.output_dir, 'test_acc.txt'), 'a')
        for round in range(self.args.rounds):
            self.printer(f"in comm round: {round}")
            self._before_train()
            party_list_this_round = self.party_list_rounds[round]

            global_w = self.global_model.state_dict()
            nets_this_round = {k: self.nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            self._local_train_net(round, nets_this_round)
            test_acc = self._global_train_net(round, nets_this_round)
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(self.global_model.state_dict(), os.path.join(self.output_dir, 'best.pth'))
            self.printer('>> Global Best Test accuracy: %f' % best_acc)
            f.write(f'{round},{test_acc}\n')

        f.close()
        self._tsne_visualize(self.global_model, self.test_dl, os.path.join(self.output_dir, 'tsne.png'))

class FedAvgOurs(FedAvg):
    ''' FedAvg+ConTre '''
    def _local_train_net(self, round, nets_this_round):
        cos = torch.nn.CosineSimilarity(dim=-1)
        for net_id, net in nets_this_round.items():
            #if net_id != 2: continue
            dataidxs = self.net_dataidx_map[net_id]
            self.printer(f'Training network {net_id}. n_training {len(dataidxs)}')
            train_dl_local, test_dl_local, _, _ = get_dataloader(self.args.dataset,
                                                                 self.args.datadir,
                                                                 self.args.batch_size,
                                                                 32,
                                                                 dataidxs)
            net.to(self.device)
            net.train()
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            for epoch in range(self.args.epochs):
                epoch_loss = []
                epoch_con_loss = []
                for batch_idx, (image, label) in enumerate(train_dl_local):
                    image = image.float().to(self.device)
                    label = label.long().to(self.device)

                    optimizer.zero_grad()
                    image.requires_grad = True
                    label.requires_grad = False
                    _, prob, out = net(image)
                    loss = criterion(out, label)
                    epoch_loss.append(loss.item())

                    # contre loss
                    con_prob = cos(prob.unsqueeze(dim=1), prob.unsqueeze(dim=0)).view(-1, 1)
                    con_label = (label.unsqueeze(dim=1) == label.unsqueeze(dim=0)).view(-1).long()
                    con_weight = 1.0 - get_current_consistency_weight(round, consistency=1.0, consistency_rampup=self.args.rounds//3)
                    con_loss = criterion(torch.cat([con_prob, 1.0-con_prob], dim=1) / 1.0, con_label) * con_weight
                    loss += con_loss
                    epoch_con_loss.append(con_loss.item())

                    loss.backward()
                    optimizer.step()
                self.printer(f'Round: {round}\tEpoch: {epoch}\tLoss: {np.mean(epoch_loss):.6f}\tConLoss: {np.mean(epoch_con_loss):.6f}\tConWeight: {con_weight:.4f}')

            test_acc, conf_matrix, _ = self._compute_accuracy(net, test_dl_local, get_confusion_matrix=True)
            self.printer(f'>> Test Accuracy: {test_acc}')
            net.to('cpu')
            self.printer(' ** Training complete **')