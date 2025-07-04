import torch
import random
import numpy as np
from torch import nn
THR1, THR2 = 1e-4, 1e-3


def score_nds(network, device, inputs, targets, args):
    network = network.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    NASWOT, MeCo, ZiCo, SSNIP = get_score1(network, inputs, targets, nn.CrossEntropyLoss().cuda(), args)
    return NASWOT, MeCo, ZiCo, SSNIP

def get_score1(net, inputs, targets, loss_fn, args):
    split_data = 2
    result_meco = []
    batch_size = len(inputs)//split_data
    net.K = np.zeros((batch_size, batch_size))
    iteration = 0

    def forward_nwot(module, inp, out):
        if iteration == 0:
            inp = inp[0].view(inp[0].size(0), -1)
            x = torch.gt(inp, 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()
        else:
            pass

    def forward_meco(module, data_input, data_output):
        if iteration == 0:
            fea = data_output[0].detach()
            n = fea.shape[0]
            if n > 3:
                fea = fea.reshape(n, -1)
                idxs = random.sample(range(n), 4)
                fea = fea[idxs, :]
                corr = torch.corrcoef(fea)
                corr.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                values = torch.linalg.eigvalsh(corr)
                result = torch.min(values) * n / 4
                result_meco.append(result.cpu().item())
        else:
            pass

    for name, module in net.named_modules():
        module.register_forward_hook(forward_meco)
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(forward_nwot)

    net.zero_grad()
    grad_dict = {}
    net.train()
    N = inputs.shape[0]
    SSNIP = 0

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        if args.ptype == 'nasbench201' or args.ptype == 'nasbenchsss':
            _, logits = net(inputs[st:en])
        else:   #nds
            logits, _ = net(inputs[st:en])

        loss = loss_fn(logits, targets[st:en])
        loss.backward()
        grad_dict, ssnip = getgrad(net, grad_dict, sp)
        SSNIP = SSNIP + ssnip
        iteration = iteration + 1
    ZiCo = caculate_zico(grad_dict)

    result_meco = np.array(result_meco)
    result_meco[np.isnan(result_meco) | np.isinf(result_meco)] = 1e-8
    MeCo = np.sum(result_meco)
    SSNIP = SSNIP.item()
    _, NASWOT = np.linalg.slogdet(net.K)
    if np.isnan(NASWOT) or np.isinf(NASWOT):
        NASWOT = 1e-8
    if np.isnan(SSNIP) or np.isinf(SSNIP):
        SSNIP = 1e-8
    if np.isnan(ZiCo) or np.isinf(ZiCo):
        ZiCo = 1e-8
    MeCo = max(1e-8, MeCo)
    ZiCo = max(1e-8, ZiCo)
    SSNIP = max(1e-8, SSNIP)

    return NASWOT, MeCo, ZiCo, SSNIP



def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    ssnip = 0
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                try:
                    mgrad = mod.weight.grad
                    if mgrad is not None:
                        grad_dict[name] = [mgrad.data.cpu().reshape(-1).numpy()]
                        v = (mgrad.data).abs()
                        ssnip += ((v >= THR1) * (v < THR2) * v * mod.weight.data).abs().sum()
                    else:
                        grad_dict[name] = [torch.zeros_like(mod.weight).cpu().numpy()]
                except Exception as e:
                    print('zico getgrad')
                    print(e)
                    continue
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                try:
                    grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy() if mod.weight.grad is not None else torch.zeros_like(mod.weight).cpu().numpy())  # add

                except Exception as e:
                    print('zico getgrad')
                    print(e)
                    continue
    return grad_dict, ssnip


def caculate_zico(grad_dict):
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum_abs = 0

    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)

        v = np.abs(grad_dict[modname])[0][nonzero_idx]
        tmp = nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]
        tmpsum = np.sum((v >= THR1) * (v < THR2) * tmp)

        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
    return nsr_mean_sum_abs
