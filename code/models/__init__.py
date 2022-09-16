import os
for name in os.listdir('models'):
    if name.startswith('_') or name.startswith('.'): continue
    name = name.replace('.py', '')
    exec(f'from models.{name} import *')

def init_nets(net_configs, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'cifar10-con'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    elif args.dataset == 'isic':
        n_classes = 4
    elif args.dataset == 'dermnet':
        n_classes = 23
    else:
        assert NotImplementedError(args.dataset)
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type
