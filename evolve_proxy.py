import argparse
import datetime
import datasets
import random
import numpy as np
import torch
import os, copy, time, logging, sys, pickle, inspect
from scipy import stats
from searchspace import searchspace
import datasets.data as data
from score_function.score import score_nds
from utils import evolve, initialize_population, save_population

tasks = ['nasbench201_cifar10', 'nasbench201_cifar100', 'nasbench201_ImageNet16-120', 'nasbenchsss_cifar10',
         'nasbenchsss_cifar100','nasbenchsss_ImageNet16-120','nds_darts', 'nds_enas', 'nds_pnas', 'nds_nasnet', 'nds_amoeba']

parser = argparse.ArgumentParser(description='ECP')
parser.add_argument('--data_loc', default='./datasets', type=str, help='dataset folder')
parser.add_argument('--api_loc', type=str, default='./APIs', help='path to API')
parser.add_argument('--save_loc', default='./results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='ECP', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--ptype', default='nds_darts', type=str, help='the nas search space to use, nds_pnas nds_enas  nds_darts nds_darts_fix-w-d nds_nasnet nds_amoeba nds_resnet nds_resnext-a nds_resnext-b')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', default=False, action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--acc_type', default='ori-test', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_runs', default=100, type=int)
parser.add_argument('--num_train_samples', default=1000, type=int)
parser.add_argument('--num_generations', default=50, type=int)
parser.add_argument('--pop_size', default=50, type=int)
parser.add_argument('--particle_length', default=4, type=int)
parser.add_argument('--save_dir', default='./results', type=str, help='folder to save results')
parser.add_argument('--nums_top_rank', default=10, type=int, help='how many top scoring networks used for rank loss calculation')
args = parser.parse_args()
args.save_dir = args.save_dir+'/populations_'+args.ptype+'_'+args.dataset+'_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def update_best_particle(population, KTau_list, gbest, pbest):

    if not pbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_ktauSet = copy.deepcopy(KTau_list)
        gbest_individual, gbest_ktau = getGbest([pbest_individuals, pbest_ktauSet])
    else:
        gbest_individual, gbest_ktau = gbest
        pbest_individuals, pbest_ktauSet = pbest

        for i, ktau in enumerate(KTau_list):
            if ktau > pbest_ktauSet[i]:
                pbest_individuals[i] = copy.deepcopy(population[i])
                pbest_ktauSet[i] = copy.deepcopy(KTau_list[i])

            if ktau > gbest_ktau:
                gbest_individual = copy.deepcopy(population[i])
                gbest_ktau = copy.deepcopy(KTau_list[i])

    return [gbest_individual, gbest_ktau], [pbest_individuals, pbest_ktauSet]

def getGbest(pbest):
    pbest_individuals, pbest_ktauSet = pbest
    gbest_ktau = 0
    gbest = None

    for i,indi in enumerate(pbest_individuals):
        if abs(pbest_ktauSet[i]) > abs(gbest_ktau):
            gbest = copy.deepcopy(indi)
            gbest_ktau = copy.deepcopy(pbest_ktauSet[i])

    return gbest, gbest_ktau

def aggregate_scores(individual, x, y, z, w):
    [a1, a2, a3, a4] = individual
    x[x == 0],y[y == 0],z[z == 0],w[w == 0] = 1e-8,1e-8,1e-8,1e-8
    scores = (a1 * np.log(x) + a2 * np.log(y) + a3 * np.log(z) + a4 * np.log(w))
    return scores

def score_calculation(inputs, targets, indices):
    search_space = searchspace.get_search_space(args)
    scores_NASWOT = np.zeros(len(indices))
    scores_MeCo = np.zeros(len(indices))
    scores_ZiCo = np.zeros(len(indices))
    scores_SSNIP = np.zeros(len(indices))
    scores_flops = np.zeros(len(indices))
    val_accs = np.zeros(len(indices))
    print('args.trainval: ', args.trainval)
    for i,ind in enumerate(indices):
        try:
            network = search_space.get_net(ind)
        except:
            try:
                network = search_space.get_network(ind)
            except Exception as e:
                print(e)
        try:
            NASWOT, MeCo, ZiCo, SSNIP = score_nds(network, device, inputs, targets, args)
            scores_NASWOT[i] = NASWOT
            scores_MeCo[i] = MeCo
            scores_ZiCo[i] = ZiCo
            scores_SSNIP[i] = SSNIP
            val_accs[i] = search_space.get_final_accuracy(ind, args.acc_type, args.trainval) #get validation acc, 200 epochs
        except Exception as e:
            print(e)
    return scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs

def score_calc_full(inputs, targets, args):
    sspace = searchspace.get_search_space(args)
    scores_NASWOT = np.zeros(len(sspace))
    scores_MeCo = np.zeros(len(sspace))
    scores_ZiCo = np.zeros(len(sspace))
    scores_SSNIP = np.zeros(len(sspace))
    scores_flops = np.zeros(len(sspace))
    test_accs = np.zeros(len(sspace))
    print('args.trainval: ', args.trainval)
    for i, (uid, network) in enumerate(sspace):
        try:
            NASWOT, MeCo, ZiCo, SSNIP = score_nds(network, device, inputs, targets, args)
            scores_NASWOT[i] = NASWOT
            scores_MeCo[i] = MeCo
            scores_ZiCo[i] = ZiCo
            scores_SSNIP[i] = SSNIP
            test_accs[i] = sspace.get_final_accuracy(uid, args.acc_type, args.trainval)
        except Exception as e:
            print(e)
    return scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, test_accs


def performance_evaluation(population, scores_proxies, task_score_proxies):
    scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs = scores_proxies

    x = scores_NASWOT
    y = scores_MeCo
    z = scores_ZiCo
    w = scores_SSNIP

    acc_ranks_inverse = stats.rankdata(val_accs)  # large accs get large ranks

    performance_list = []
    for particle in population:
        scores = aggregate_scores(particle, x, y, z, w)
        tau, p = stats.kendalltau(val_accs, scores)
        performance_list.append(tau)
        logging.info('[Main Task: %s, KTau: %.3f]' % (init_ptype + '_' + init_dataset, tau))
    return performance_list

def obtain_test_performance(indi, scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, test_accs):

    x = scores_NASWOT
    y = scores_MeCo
    z = scores_ZiCo
    w = scores_SSNIP

    acc_ranks_inverse = stats.rankdata(test_accs)  # large accs get large ranks

    scores = aggregate_scores(indi, x, y, z, w)

    final_ktau, p = stats.kendalltau(test_accs, scores)
    final_rho, pp = stats.spearmanr(test_accs, scores)

    top_indices = np.argsort(-scores)[:args.nums_top_rank]  # small to large
    top_score_inverse_ranks = acc_ranks_inverse[top_indices]
    avg_top_rank = len(scores) - np.mean(top_score_inverse_ranks)

    return final_ktau, final_rho, avg_top_rank, np.mean(test_accs[top_indices]), np.std(test_accs[top_indices])

def configuration_setup(task_, istrain):
    # setting the args parameters
    if task_.__contains__('nasbench'):
        ptype, dset = task_.split('_')
        args.ptype = ptype
        args.dataset = dset
        if args.dataset == 'cifar10':
            args.data_loc = './datasets/CIFAR10_data/'
            if istrain:
                args.acc_type = 'x-valid'
            else:
                args.acc_type = 'ori-test'
        elif args.dataset == 'cifar100':
            args.data_loc = './datasets/CIFAR100_data/'
            if istrain:
                args.acc_type = 'x-valid'
            else:
                args.acc_type = 'x-test'
        else:
            args.data_loc = './datasets/ImageNet16/'
            if istrain:
                args.acc_type = 'x-valid'
            else:
                args.acc_type = 'x-test'
    else:
        args.ptype = task_
        args.dataset = 'cifar10'
        args.data_loc = './datasets/CIFAR10_data/'
        if istrain:
            args.acc_type = 'x-valid'
        else:
            args.acc_type = 'x-test'

def performance_test(indi):
    args.trainval = False
    final_ktaus, final_rhos = [], []
    logging.info('Target task: [%s_%s]', args.ptype, args.dataset)
    logging.info(inspect.getsource(aggregate_scores))
    for task in tasks:
        configuration_setup(task, False)
        if task.__contains__('nasbench'):
            if os.path.isfile('./performance_set_%d_%d.pkl' % (args.batch_size, args.seed)):
                with open('./performance_set_%d_%d.pkl' % (args.batch_size, args.seed), 'rb') as f:
                    performance_set = pickle.load(f)
                [scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs, test_accs] = performance_set[task]
            else:
                train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
                data_iterator = iter(train_loader)
                inputs_, targets_ = next(data_iterator)
                # for nasbench, evaluated on val acc, test on all architectures
                scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, test_accs = score_calc_full(inputs_, targets_, args)

        else:
            if os.path.isfile('./performance_set_%d_%d.pkl'%(args.batch_size, args.seed)) and init_ptype.__contains__('nasbench'):
                with open('./performance_set_%d_%d.pkl'%(args.batch_size, args.seed), 'rb') as f:
                    performance_set = pickle.load(f)
                [scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs, test_accs] = performance_set[task]
            else:
                train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
                data_iterator = iter(train_loader)
                inputs_, targets_ = next(data_iterator)
                # for nds, eval on test acc of train indices,test on test acc of test indices (rest of the architectures)
                scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, test_accs = score_calculation(inputs_, targets_, test_indices)

        ktau_task, rho_task, avg_top_rank, mean_top_acc, std_top_acc = obtain_test_performance(indi, scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, test_accs)
        # logging.info("on task [%s], Test KTau: %.3f, Test Rho: %.3f", task, ktau_task, rho_task)
        logging.info("on task [%s], Test KTau: %.3f, Test Rho: %.3f, avg_top%d_rank: %.3f, mean_top%d_acc: %.3f, std_top%d_acc: %.3f",
            task, ktau_task, rho_task, args.nums_top_rank, avg_top_rank, args.nums_top_rank, mean_top_acc, args.nums_top_rank, std_top_acc)

        final_ktaus.append(ktau_task)
        final_rhos.append(rho_task)
    return final_ktaus, final_rhos

def main():
    gen_no = 0
    start = time.time()

    data_iterator = iter(train_loader)
    inputs, targets = next(data_iterator)

    args.trainval = True
    scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs = score_calculation(inputs, targets, train_indices)
    scores_proxies = [scores_NASWOT, scores_MeCo, scores_ZiCo, scores_SSNIP, scores_flops, val_accs]

    task_score_proxies = {}

    # goes back to target task setup
    if init_ptype.__contains__('nasbench'):
        configuration_setup(init_ptype+'_'+init_dataset, istrain=True)
    else:
        configuration_setup(init_ptype, istrain=True)

    population = initialize_population(args)

    KTau_list = performance_evaluation(population, scores_proxies, task_score_proxies)
    [gbest_individual, gbest_ktau], [pbest_individuals, pbest_ktauSet] = update_best_particle(population, KTau_list, gbest=None, pbest=None)

    save_population('population', population, KTau_list, args, gen_no)
    save_population('pbest', pbest_individuals, pbest_ktauSet, args, gen_no)
    save_population('gbest', [gbest_individual], [gbest_ktau], args, gen_no)

    gen_no += 1
    velocity_set = []
    for ii in range(len(population)):
        velocity_set.append([0.0] * len(population[ii]))

    for curr_gen in range(gen_no, args.num_generations):
        args.curr_gen = curr_gen
        logging.info('EVOLVE[%d-gen]-Begin pso evolution', curr_gen)
        population, velocity_set = evolve(population, gbest_individual, pbest_individuals, velocity_set, args)
        logging.info('EVOLVE[%d-gen]-Finish pso evolution', curr_gen)

        logging.info('EVOLVE[%d-gen]-Begin to evaluate the fitness', curr_gen)
        KTau_list = performance_evaluation(population, scores_proxies, task_score_proxies)
        logging.info('EVOLVE[%d-gen]-Finish the evaluation', curr_gen)
        [gbest_individual, gbest_ktau], [pbest_individuals, pbest_ktauSet] = update_best_particle(population, KTau_list, gbest=[gbest_individual, gbest_ktau], pbest=[pbest_individuals, pbest_ktauSet])
        logging.info('EVOLVE[%d-gen]-Finish the updating', curr_gen)
        logging.info("GBest: %s, KTau: %.4f", gbest_individual, gbest_ktau)

        save_population('population', population, KTau_list, args, curr_gen)
        save_population('pbest', pbest_individuals, pbest_ktauSet, args, curr_gen)
        save_population('gbest', [gbest_individual], [gbest_ktau], args, curr_gen)

    end = time.time()
    logging.info('Total Search Time: %.2f seconds', (end - start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    logging.info("%02dh:%02dm:%02ds", h, m, s)

    search_time = str("%02dh:%02dm:%02ds" % (h, m, s))
    logging.info('Begin to test the gBest')
    final_ktau, final_rho = performance_test(gbest_individual)
    save_population('final_test', [gbest_individual], final_ktau, args, -1, gbest_ktau, search_time, final_rho)


def train_test_split(space_size):
    count = args.num_train_samples
    train_indices = [random.randint(0, space_size) for _ in range(count)]
    test_indices = list(set(range(space_size))-set(train_indices))
    random.shuffle(test_indices)

    return train_indices, test_indices

def create_directory():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

if __name__ == '__main__':
    create_directory()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    savedataset = args.dataset

    if args.dataset == 'cifar10':
        args.data_loc = './datasets/CIFAR10_data/'
        args.acc_type = 'x-valid'
    elif args.dataset == 'cifar100':
        args.data_loc = './datasets/CIFAR100_data/'
        args.acc_type = 'x-valid'
    else:
        args.data_loc = './datasets/ImageNet16/'
        args.acc_type = 'x-valid'
    init_ptype = copy.deepcopy(args.ptype)
    init_dataset = copy.deepcopy(args.dataset)
    search_space = searchspace.get_search_space(args)
    train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
    os.makedirs(args.save_loc, exist_ok=True)

    filename = f'{args.save_loc}/{args.save_string}_{args.ptype}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.trainval}_{args.batch_size}_{args.seed}'
    accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.ptype}_{savedataset}_{args.trainval}'

    train_indices, test_indices = train_test_split(len(search_space))

    main()
