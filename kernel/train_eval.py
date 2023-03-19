import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb

from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs


def cross_validation_with_val_set(dataset,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device,
                                  logger=None,
                                  metric = 'acc',
                                  local_rank = None,
                                  world_size = None):

    final_train_losses, val_losses, accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            if not torch.distributed.is_initialized():
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            else:
                if local_rank is None or world_size is None:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                else:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=local_rank)
                train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler = train_sampler, pin_memory = True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        if torch.distributed.is_initialized():
            model.module.reset_parameters()
        else:
            model.to(device).reset_parameters()
        #model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val_losses = []
        cur_accs = []
        for epoch in pbar:
            if torch.distributed.is_initialized():
                train_sampler.set_epoch(epoch)
            train_loss = train(model, optimizer, train_loader, device)
            cur_val_losses.append(eval_loss(model, val_loader, device))
            cur_accs.append(eval_acc(model, test_loader, device, metric = metric))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': cur_val_losses[-1],
                'test_acc': cur_accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        val_losses += cur_val_losses
        accs += cur_accs

        loss, argmin = tensor(cur_val_losses).min(dim=0)
        acc = cur_accs[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
            fold, eval_info["train_loss"], loss, acc
        )
        print(log)
        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    #average_train_loss = float(np.mean(final_train_losses))
    #std_train_loss = float(np.std(final_train_losses))

    log = 'Val Loss: {:.4f}, Test Accuracy: {:.3f} + {:.3f}, Duration: {:.3f}'.format(
        loss.mean().item(),
        acc.mean().item(),
        acc.std().item(),
        duration.mean().item()
    ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    print(log)
    if logger is not None:
        logger(log)

    return loss.mean().item(), acc.mean().item(), acc.std().item()


def cross_validation_without_val_set( dataset,
                                      model,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      weight_decay,
                                      device, 
                                      logger=None,
                                      metric = 'acc'):

    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            test_losses.append(eval_loss(model, test_loader, device))
            accs.append(eval_acc(model, test_loader, device, metric = metric))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} + {:.3f}, ' + 
          'Test Final Accuracy: {:.3f} + {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    print(log)
    if logger is not None:
        logger(log)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold2(dataset, folds):
    kf = KFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, test_idx in kf.split(dataset):
        test_indices.append(torch.from_numpy(test_idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device, accelerator = None):
    model.train()

    total_loss = 0
    for data in loader:
        #print("Node number:{}, Edge number: {}".format(len(data.x), data.edge_index.size()[1]))
        if len(data.x) > 150000 and data.edge_index.size()[1] > 3000000:
            # in case of OOM (can delete for large memory GPUs)
            continue
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()


    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device, metric = 'acc'):
    model.eval()

    correct = 0
    out = []; true = []
    for data in loader:
        data = data.to(device)
        if metric == 'acc':
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        else:
            out += [model(data).cpu()]
            true += [data.y.cpu()]
    if metric == 'roc':
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(true, out)
    elif metric == 'ap':
        from sklearn.metrics import average_precision_score
        score = average_precision_score(true, out)
    elif metric != 'acc':
        print("metric not implemented")

    if metric == 'acc':
        return correct / len(loader.dataset)
    else:
        return score



def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def train_cycle(model, optimizer, data, index, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data)[index]
    true = data.cycles[index].float()
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss = bce_loss(out, true)
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    return total_loss

def eval_cycle(model, data, index, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data)[index]
        true = data.cycles[index].float()
        from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
        pred_int = (out > 0.5).long().cpu()
        out = out.cpu()
        true = true.cpu()
    return accuracy_score(true, pred_int), roc_auc_score(true, out), average_precision_score(true, out)

def train_val_cycles(dataset,
                                  model,
                                  split_ratio,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device,
                                  logger=None,
                                    seed = 1234):
    import random
    random.seed(seed)
    final_train_losses, val_losses, accs, durations = [], [], [], []
    for data in dataset:
        # only one data actually
        try:
            node_num = len(data.node_id)
        except BaseException:
            node_num = len(data.x)
        try:
            data.cycles = torch.load("/data1/count_cycles/" + dataset.name + ".pt")
            data.cycles[data.cycles != 0] = 1
        except BaseException:
            print(dataset.name + " no cycles loaded")
        shuffle_node_index = np.arange(node_num)
        random.shuffle(shuffle_node_index)
        train_split, valid_split = int(split_ratio * node_num), int((split_ratio + 1 )/ 2 * node_num)
        train_idx, valid_idx, test_idx = shuffle_node_index[:train_split], shuffle_node_index[train_split:valid_split], shuffle_node_index[valid_split:]
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val = []
        cur_test = []
        for epoch in pbar:
            train_loss = train_cycle(model, optimizer, data, train_idx, device)
            cur_val.append(eval_cycle(model, data, valid_idx, device)[2])
            cur_test.append(eval_cycle(model, data, test_idx, device))
            eval_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_ap': cur_val[-1],
                'test_acc': cur_test[-1][0],
                'test_roc': cur_test[-1][1],
                'test_ap': cur_test[-1][2]
            }
            log = 'train_loss: %0.4f, val_ap: %0.4f, test_ap: %0.4f' % (
                eval_info["train_loss"], eval_info["val_ap"], eval_info["test_ap"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        val_losses += cur_val
        accs += cur_test

        loss, argmin = tensor(cur_val).max(dim=0)
        acc = cur_test[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Final train_loss: %0.4f, best val_ap: %0.4f, test_acc: %0.4f, test_roc: %0.4f, test_ap: %0.4f,' % (
            eval_info["train_loss"], loss, acc[0], acc[1], acc[2]
        )
        print(log)
        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)



    return loss.item(), acc[0], acc[1], acc[2]


def train_cycle_regression(model, optimizer, data, index, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    data = data.to(device)
    #out = model(data)[index]
    out, ys = model(data)
    out = out[index]

    true = data.cycles[index].float()
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(out, true)

    for i in range(len(ys)):
        loss += mse_loss(ys[i][index], true[:, :(2*i+1)]) / 10

    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    return total_loss

def eval_cycle_regression(model, data, index, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data)[0][index]
        true = data.cycles[index].float()
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        out = out.cpu()
        true = true.cpu()
    return  mean_squared_error(true, out, squared=True), mean_absolute_error(true, out), mean_squared_error(true, out, squared=False)

def train_val_cycles_regression(dataset,
                                  model,
                                  split_ratio,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device,
                                  logger=None,
                                    seed = 1234):
    import random
    random.seed(seed)
    final_train_losses, val_losses, accs, durations = [], [], [], []
    for data in dataset:
        # only one data actually
        try:
            node_num = len(data.node_id)
        except BaseException:
            node_num = len(data.x)
        try:
            data.cycles = torch.load("/data1/count_cycles/" + dataset.name + ".pt")
            #data.cycles = data.cycles[:, :3]
        except BaseException:
            print(dataset.name + " no cycles loaded")
        shuffle_node_index = np.arange(node_num)
        random.shuffle(shuffle_node_index)
        train_split, valid_split = int(split_ratio * node_num), int((split_ratio + 1 )/ 2 * node_num)
        train_idx, valid_idx, test_idx = shuffle_node_index[:train_split], shuffle_node_index[train_split:valid_split], shuffle_node_index[valid_split:]
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val = []
        cur_test = []
        for epoch in pbar:
            train_loss = train_cycle_regression(model, optimizer, data, train_idx, device)
            cur_val.append(eval_cycle_regression(model, data, valid_idx, device)[1])
            cur_test.append(eval_cycle_regression(model, data, test_idx, device))
            eval_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mae': cur_val[-1],
                'test_mse': cur_test[-1][0],
                'test_mae': cur_test[-1][1],
                'test_rmse': cur_test[-1][2]
            }
            log = 'train_loss: %0.4f, val_mae: %0.4f, test_mae: %0.4f' % (
                eval_info["train_loss"], eval_info["val_mae"], eval_info["test_mae"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        val_losses += cur_val
        accs += cur_test

        loss, argmin = tensor(cur_val).min(dim=0)
        acc = cur_test[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Final train_loss: %0.4f, best val_mae: %0.4f, test_mse: %0.4f, test_mae: %0.4f, test_rmse: %0.4f,' % (
            eval_info["train_loss"], loss, acc[0], acc[1], acc[2]
        )
        print(log)
        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)



    return loss.item(), acc[0], acc[1], acc[2]



def train_cycle_regression_GC(model, optimizer, dataloader, dataset_cycles, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    cnt_total = 0
    for cnt_b, data in enumerate(dataloader):
        data = data.to(device)
        out, ys = model(data)
        batch_graphs = num_graphs(data)
        true = torch.cat(dataset_cycles[cnt_total: cnt_total + batch_graphs], dim = 0).float().to(device)
        true = true[:, : out.size()[1]]
        cnt_total += batch_graphs

        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(out, true)

        for i in range(len(ys)):
            loss += mse_loss(ys[i], true[:, :(2*i+1)]) / len(ys)

        loss.backward()
        total_loss += loss.item() * batch_graphs
        optimizer.step()
    return total_loss / len(dataloader.dataset)

def eval_cycle_regression_GC(model, dataloader, dataset_cycles, device):
    model.eval()
    with torch.no_grad():
        for cnt_b, data in enumerate(dataloader):
            data = data.to(device)
            out, _ = model(data)
            total_out = out if cnt_b == 0 else torch.cat((total_out, out), dim = 0)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    out = total_out.cpu()
    true = torch.cat(dataset_cycles, dim=0).float()
    true = true[:, : out.size()[1]]
    #print(out)
    #print(true)
    return  mean_squared_error(true, out, squared=True), mean_absolute_error(true, out), mean_squared_error(true, out, squared=False)

def train_val_cycles_regression_GC(dataset,
                                  model,
                                  split_ratio,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device,
                                  logger=None,
                                    seed = 1234):
    import random
    random.seed(seed)
    graph_num = len(dataset)
    try:
        dataset_cycles = torch.load("/data1/count_cycles/" + dataset.name + ".pt")
        # dataset_cycles = dataset_cycles[:, :3]
    except BaseException:
        dataset_cycles = torch.load("/data1/count_cycles/ZINC.pt") # ZINC does not have name
        print("No dataset name, use ZINC instead")
    shuffle_graph_index = np.arange(graph_num)
    random.shuffle(shuffle_graph_index)
    train_split, valid_split = int(split_ratio * graph_num), int((split_ratio + 1) / 2 * graph_num)
    train_idx, valid_idx, test_idx = shuffle_graph_index[:train_split], shuffle_graph_index[
                                                                       train_split:valid_split], shuffle_graph_index[
                                                                                                 valid_split:]
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if 'adj' in dataset[0]:
        val_loader = DenseLoader(dataset[valid_idx], batch_size, shuffle=False)
        test_loader = DenseLoader(dataset[test_idx], batch_size, shuffle=False)
    else:
        val_loader = DataLoader(dataset[valid_idx], batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size, shuffle=False)

    t_start = time.perf_counter()
    pbar = tqdm(range(1, epochs + 1), ncols=70)
    cur_val = []
    cur_test = []
    for epoch in pbar:
        # shuffle the graph idx to suit the cycles idx
        random.shuffle(train_idx)
        if 'adj' in dataset[0]:
            train_loader = DenseLoader(dataset[train_idx], batch_size, shuffle=False)
        else:
            train_loader = DataLoader(dataset[train_idx], batch_size, shuffle=False)
        train_loss = train_cycle_regression_GC(model, optimizer, train_loader, [dataset_cycles[id] for id in train_idx], device)
        cur_val.append(eval_cycle_regression_GC(model, val_loader, [dataset_cycles[id] for id in valid_idx], device)[1])
        cur_test.append(eval_cycle_regression_GC(model, test_loader, [dataset_cycles[id] for id in test_idx], device))
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': cur_val[-1],
            'test_mse': cur_test[-1][0],
            'test_mae': cur_test[-1][1],
            'test_rmse': cur_test[-1][2]
        }
        log = 'train_loss: %0.4f, val_mae: %0.4f, test_mae: %0.4f' % (
            eval_info["train_loss"], eval_info["val_mae"], eval_info["test_mae"]
        )
        pbar.set_description(log)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    loss, argmin = tensor(cur_val).min(dim=0)
    acc = cur_test[argmin.item()]
    log = 'Final train_loss: %0.4f, best val_mae: %0.4f, test_mse: %0.4f, test_mae: %0.4f, test_rmse: %0.4f,' % (
        eval_info["train_loss"], loss, acc[0], acc[1], acc[2]
    )
    print(log)
    if logger is not None:
        logger(log)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()



    return loss.item(), acc[0], acc[1], acc[2]