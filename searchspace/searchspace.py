#NasBench201
from .nas_201_api import NASBench201API as API201
from .models import get_cell_based_tiny_net
#NATSBench-sss
from .nats_api import create as create_sss

#NDS
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import json, torch, random

import pandas as pd
import itertools
import numpy as np

class NASBENCH201:
    '''201'''
    def __init__(self, dataset,args):
        self.dataset = dataset
        print("Loading api...")#NAS-Bench-201-v1_0-e61699.pth
        self.api = API201('./APIs/NAS-Bench-201-v1_1-096897.pth',verbose=False)
        print("Finished loading.")
        self.operations = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3' ]
        self.args=args
    def __len__(self):
        return 15625

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_net(uid)
            yield uid, network

    def __getitem__(self, index):
        return index

    def get_net(self,index):
        index = self.api.query_index_by_arch(index)
        if self.dataset == "cifar10":
            dataname = "cifar10-valid"
            config = self.api.get_net_config(index, dataname)
            config['num_classes'] = 10
        else:
            dataname = self.dataset
            config = self.api.get_net_config(index, dataname)
            if self.dataset == 'cifar100':
                config['num_classes'] = 100
            else:
                config['num_classes'] = 120

        network = get_cell_based_tiny_net(config)
        return network
    
    def get_acc_all(self,index,args):
        index = self.api.query_index_by_arch(index)
        information = self.api.arch2infos_dict[index]['200']

        valid_info_cifar10 = information.get_metrics('cifar10-valid', 'x-valid')
        valid_acc_cifar10 = valid_info_cifar10['accuracy']
        test__info_cifar10 = information.get_metrics('cifar10', 'ori-test')
        test_acc_cifar10 = test__info_cifar10['accuracy']

        valid_info_cifar100 = information.get_metrics('cifar100', 'x-valid')
        test__info_cifar100 = information.get_metrics('cifar100', 'x-test')
        valid_acc_cifar100 = valid_info_cifar100['accuracy']
        test_acc_cifar100 = test__info_cifar100['accuracy']

        valid_info_imagenet = information.get_metrics('ImageNet16-120', 'x-valid')
        test__info_imagenet = information.get_metrics('ImageNet16-120', 'x-test')
        valid_acc_imagenet = valid_info_imagenet['accuracy']
        test_acc_imagenet = test__info_imagenet['accuracy']

        return test_acc_cifar10,valid_acc_cifar10,test_acc_cifar100,valid_acc_cifar100,test_acc_imagenet,valid_acc_imagenet

    def get_acc(self, index, args, hp='200'):
        index = self.api.query_index_by_arch(index)
        information = self.api.arch2infos_dict[index][hp]
        if args.dataset == 'cifar10':
            valid_info = information.get_metrics('cifar10-valid', 'x-valid')
            valid_acc = valid_info['accuracy']
            test__info = information.get_metrics('cifar10', 'ori-test')
            test_acc = test__info['accuracy']
        else:
            valid_info = information.get_metrics(args.dataset, 'x-valid')
            test__info = information.get_metrics(args.dataset, 'x-test')
            valid_acc = valid_info['accuracy']
            test_acc = test__info['accuracy']
        return test_acc,valid_acc

    def get_acc_by_code(self,code,args,hp='200'):
        if hp is not str:
            hp = str(hp)
        index = self.get_index_by_code(code,args)
        information = self.api.arch2infos_dict[index][hp]
        if args.dataset == 'cifar10':
            valid_info = information.get_metrics('cifar10-valid', 'x-valid')
            valid_acc = valid_info['accuracy']
            test__info = information.get_metrics('cifar10', 'ori-test')
            test_acc = test__info['accuracy']
        else:
            valid_info = information.get_metrics(args.dataset, 'x-valid')
            test__info = information.get_metrics(args.dataset, 'x-test')
            valid_acc = valid_info['accuracy']
            test_acc = test__info['accuracy']
        return test_acc,valid_acc

    def get_acc_by_code_all(self,code,args):
        index = self.get_index_by_code(code,args)
        information = self.api.arch2infos_dict[index]['200']

        valid_info_cifar10 = information.get_metrics('cifar10-valid', 'x-valid')
        valid_acc_cifar10 = valid_info_cifar10['accuracy']
        test__info_cifar10 = information.get_metrics('cifar10', 'ori-test')
        test_acc_cifar10 = test__info_cifar10['accuracy']

        valid_info_cifar100 = information.get_metrics('cifar100', 'x-valid')
        test__info_cifar100 = information.get_metrics('cifar100', 'x-test')
        valid_acc_cifar100 = valid_info_cifar100['accuracy']
        test_acc_cifar100 = test__info_cifar100['accuracy']

        valid_info_imagenet = information.get_metrics('ImageNet16-120', 'x-valid')
        test__info_imagenet = information.get_metrics('ImageNet16-120', 'x-test')
        valid_acc_imagenet = valid_info_imagenet['accuracy']
        test_acc_imagenet = test__info_imagenet['accuracy']

        return test_acc_cifar10,valid_acc_cifar10,test_acc_cifar100,valid_acc_cifar100,test_acc_imagenet,valid_acc_imagenet

    def get_index_by_code(self,code,args):
        node_str = ""
        base=-0
        for j in range(1,4):
            node_str += '|'
            for k in range(0,j):
                node_str = node_str + self.operations[int(code[base])] + '~'+ str(k) +'|'
                base+=1
            node_str += '+'
        node_str = node_str[0:-1]
        index = self.api.query_index_by_arch(node_str)
        return index
    
    def get_net_by_code(self,code,args):
        index = self.get_index_by_code(code,args)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset

        config = self.api.get_net_config(index, dataname)
        if self.dataset == "cifar10":
            config['num_classes'] = 10
        else:
            if self.dataset == 'cifar100':
                config['num_classes'] = 100
            else:
                config['num_classes'] = 120
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_training_time_by_code(self,code,args,hp):
        index = self.get_index_by_code(code,args)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        info = self.api.get_more_info(
            index, dataname, iepoch=None, hp=hp, is_random=True
        )
        time_cost = info["train-all-time"] + info["valid-per-time"]
        
        return time_cost

    def get_training_time(self,index,args,hp):
        index = self.api.query_index_by_arch(index)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        info = self.api.get_more_info(
            index, dataname, iepoch=None, hp=hp, is_random=True
        )
        time_cost = info["train-all-time"] + info["valid-per-time"]
        
        return time_cost

    def get_complexity(self, index):
        # arch_info = self.arch2infos_dict[arch_index][hp]
        arch_info = self.api.arch2infos_dict[index]['200']
        info = arch_info.get_compute_costs(self.dataset)  # the information of costs
        flops, params, latency = info['flops'], info['params'], info['latency']
        return flops/1e6, params/1e6

    # for NDS
    def get_final_accuracy(self, uid, acc_type, trainval):
        #archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
            #info = self.api.query_by_index(uid, 'cifar10-valid', hp='200')
            #info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp='200', is_random=True)
        else:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics(self.dataset, acc_type)
            #info = self.api.query_by_index(uid, self.dataset, hp='200')
            #info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp='200', is_random=True)
        return info['accuracy']

    def get_valid_accuracy(self, uid, acc_type, hp=12, trainval=True):
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp=str(hp)).get_metrics('cifar10-valid', 'x-valid')
        else:
            info = self.api.query_meta_info_by_index(uid, hp=str(hp)).get_metrics(self.dataset, acc_type)
        return info['accuracy']


class NATSBENCHSSS:
    '''sss'''
    def __init__(self, dataset, args):
        self.dataset = dataset
        print("Loading api...")
        self.api = create_sss("./APIs/NATS-sss-v1_0-50262-simple/", 'sss', fast_mode=True, verbose=False)
        print("Finished loading.")
        self.args=args

    def __len__(self):
        return 32768

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_net(uid)
            yield uid, network

    def __getitem__(self, index):
        return index


    def get_net(self,index):
        index = self.api.query_index_by_arch(index)
        if self.args.dataset == "cifar10":
            dataname = "cifar10-valid"
            config = self.api.get_net_config(index, dataname)
            config['num_classes'] = 10
        else:
            dataname = self.args.dataset
            config = self.api.get_net_config(index, dataname)
            if self.args.dataset == "cifar100":
                config['num_classes'] = 100
            else:
                config['num_classes'] = 120

        network = get_cell_based_tiny_net(config)
        return network
    
    def get_acc(self, index, args,hp=90):
        index = self.api.query_index_by_arch(index)
        info = self.api.get_more_info(index, args.dataset, hp= hp)
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return info['test-accuracy'],validation_accuracy

    def get_acc_all(self, index, args):
        info_cifar10 = self.api.get_more_info(index, 'cifar10', hp=90)
        info_cifar100 = self.api.get_more_info(index, 'cifar100', hp=90)
        info_imagenet = self.api.get_more_info(index, 'ImageNet16-120', hp=90)

        validation_accuracy_cifar10, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar10', hp=90)
        validation_accuracy_cifar100, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar100', hp=90)
        validation_accuracy_imagenet, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='ImageNet16-120', hp=90)        
        return info_cifar10['test-accuracy'],validation_accuracy_cifar10,info_cifar100['test-accuracy'],validation_accuracy_cifar100,info_imagenet['test-accuracy'],validation_accuracy_imagenet

    def get_training_time(self,index,args,hp):
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return time_cost

    def get_complexity(self, index):
        # arch_info = self.arch2infos_dict[arch_index][hp]
        arch_info = self.api.arch2infos_dict[index]['90']
        info = arch_info.get_compute_costs(self.dataset)  # the information of costs
        flops, params, latency = info['flops'], info['params'], info['latency']
        return flops/1e6, params/1e6

    # for NDS
    def get_final_accuracy(self, uid, acc_type, trainval):
        # archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='90').get_metrics('cifar10-valid', 'x-valid')
            # info = self.api.query_by_index(uid, 'cifar10-valid', hp='200')
            # info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp='200', is_random=True)
        else:
            info = self.api.query_meta_info_by_index(uid, hp='90').get_metrics(self.dataset, acc_type)
            # info = self.api.query_by_index(uid, self.dataset, hp='200')
            # info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp='200', is_random=True)
        return info['accuracy']

    def get_valid_accuracy(self, uid, acc_type, hp=12, trainval=True):
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp=str(hp)).get_metrics('cifar10-valid', 'x-valid')
        else:
            info = self.api.query_meta_info_by_index(uid, hp=str(hp)).get_metrics(self.dataset, acc_type)
        return info['accuracy']

######################################3
# NDS

class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x), x

def return_feature_layer(network, prefix=''):
    # for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')
class NDS:
    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'./APIs/nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        #print(config)
        if 'genotype' in config:
            #print('geno')
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1000, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(config['width'], 10, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.
            #print(config)
            #print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            #"res_stem_cifar": ResStemCifar,
            #"res_stem_in": ResStemIN,
            #"simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return_feature_layer(network)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def get_complexity(self, uid):
        netinfo = self.data[uid]
        return netinfo['flops']/1e6, netinfo['params']/1e6

    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def get_final_accuracy(self, uid, acc_type, trainval):
        return 100.-self.data[uid]['test_ep_top1'][-1]

def get_search_space(args):
    if args.ptype == 'nasbench201':
        return NASBENCH201(args.dataset, args)
    elif args.ptype == 'nds_resnet':
        return NDS('ResNet')
    elif args.ptype == 'nds_amoeba':
        return NDS('Amoeba')
    elif args.ptype == 'nds_amoeba_in':
        return NDS('Amoeba_in')
    elif args.ptype == 'nds_darts_in':
        return NDS('DARTS_in')
    elif args.ptype == 'nds_darts':
        return NDS('DARTS')
    elif args.ptype == 'nds_darts_fix-w-d':
        return NDS('DARTS_fix-w-d')
    elif args.ptype == 'nds_darts_lr-wd':
        return NDS('DARTS_lr-wd')
    elif args.ptype == 'nds_enas':
        return NDS('ENAS')
    elif args.ptype == 'nds_enas_in':
        return NDS('ENAS_in')
    elif args.ptype == 'nds_enas_fix-w-d':
        return NDS('ENAS_fix-w-d')
    elif args.ptype == 'nds_pnas':
        return NDS('PNAS')
    elif args.ptype == 'nds_pnas_fix-w-d':
        return NDS('PNAS_fix-w-d')
    elif args.ptype == 'nds_pnas_in':
        return NDS('PNAS_in')
    elif args.ptype == 'nds_nasnet':
        return NDS('NASNet')
    elif args.ptype == 'nds_nasnet_in':
        return NDS('NASNet_in')
    elif args.ptype == 'nds_resnext-a':
        return NDS('ResNeXt-A')
    elif args.ptype == 'nds_resnext-a_in':
        return NDS('ResNeXt-A_in')
    elif args.ptype == 'nds_resnext-b':
        return NDS('ResNeXt-B')
    elif args.ptype == 'nds_resnext-b_in':
        return NDS('ResNeXt-B_in')
    elif args.ptype == 'nds_vanilla':
        return NDS('Vanilla')
    elif args.ptype == 'nds_vanilla_lr-wd':
        return NDS('Vanilla_lr-wd')
    elif args.ptype == 'nds_vanilla_lr-wd_in':
        return NDS('Vanilla_lr-wd_in')
    elif args.ptype == 'nasbenchsss':
        return NATSBENCHSSS(args.dataset, args)
