import os
import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, GINEConv
import torch_geometric.transforms as T
from GraphCountDataset import dataset_random_graph
import os.path as osp
import os, sys
from shutil import copy, rmtree
import pdb
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, GELU, BatchNorm1d as BN, Dropout
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool, GATConv
from utils_edge_efficient import create_subgraphs
from sklearn.metrics import mean_absolute_error as MAE
from modules.ppgn_modules import *
from torch_geometric.utils import degree, dropout_adj, to_dense_batch, to_dense_adj

def MyTransform(data):
    data.y = data.y[:, int(args.target)]
    return data

class NestedGIN_eff(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False, use_cycle=False, graph_pred=True,
                 use_id=None, dropout=0.2, multi_layer=False, edge_nest=False):
        super(NestedGIN_eff, self).__init__()
        self.use_rd = use_rd
        self.use_z = True
        self.graph_pred = graph_pred  # delete the final graph-level pooling
        self.use_cycle = use_cycle  # to mark whther predicting the cycle or not
        self.use_id = use_id
        self.dropout = dropout  # dropout 0.1 for multilayer, 0.2 for no multi
        self.multi_layer = multi_layer  # to use multi layer supervision or not
        self.edge_nest = edge_nest  # denote whether using the edge-level nested information
        z_in = 1800# if self.use_rd else 1700
        emb_dim = hidden
        self.z_initial = torch.nn.Embedding(z_in, emb_dim)
        self.z_embedding = Sequential(Dropout(dropout),
                                      torch.nn.BatchNorm1d(emb_dim),
                                      ReLU(),
                                      Linear(emb_dim, emb_dim),
                                      Dropout(dropout),
                                      torch.nn.BatchNorm1d(emb_dim),
                                      ReLU()
                                      )
        input_dim = 10#1800#dataset.num_features
        #if self.use_z or self.use_rd:
        #    input_dim += 8
        self.x_embedding = Sequential(Linear(input_dim, hidden),
                           Dropout(dropout),
                           BN(hidden),
                           ReLU(),
                           Linear(hidden, hidden),
                           Dropout(dropout),
                           BN(hidden),
                           ReLU()
        )
        if use_id is None:
            # self.conv1 = GCNConv(input_dim, hidden)

            self.conv1 = GINEConv(
                Sequential(
                    Linear(input_dim, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=True,
                edge_dim = hidden)

            #self.conv1 = GATConv(input_dim, hidden, edge_dim = hidden, add_self_loops = False)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if use_id is None:

                self.convs.append(GINEConv(
                    Sequential(
                        Linear(hidden, hidden),
                        Dropout(dropout),
                        BN(hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        Dropout(dropout),
                        BN(hidden),
                        ReLU(),
                    ),
                    train_eps=True,
                    edge_dim = hidden))

                #self.convs.append(GATConv(hidden, hidden, edge_dim = hidden, add_self_loops = False))

        self.lin1 = torch.nn.Linear(num_layers * hidden + hidden, hidden)
        #self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        #self.lin1 = torch.nn.Linear(hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not use_cycle:
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 1)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for layer in self.z_embedding.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        data.to(self.lin1.weight.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        #edge_pos[:, :200] = 0
        #edge_pos[:, 200:500] = 0
        #edge_pos[:, -1300:] = 0
        
        if hasattr(data, 'edge_pos'):
            # original, slow version
            edge_pos = data.edge_pos.float()
            z_emb = torch.mm(edge_pos, self.z_initial.weight)
        else:
            # new, fast version
            
            # for ablation study
            #mask_index = (data.pos_index >= 500)
            #mask_index = torch.logical_and((data.pos_index >= 200), (data.pos_index < 500))
            #mask_index = (data.pos_index < 500)
            #z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index[~mask_index]], data.pos_enc[~mask_index].view(-1, 1)), data.pos_batch[~mask_index])
            
            z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index], data.pos_enc.view(-1, 1)), data.pos_batch)
        z_emb = self.z_embedding(z_emb)
        
        #z_emb = self.z_embedding(edge_pos)

        if self.use_id is None:
            x = self.conv1(x, edge_index, z_emb)
        else:
            x = self.conv1(x, edge_index, data.node_id)

        #xs = [x]
        xs = [self.x_embedding(data.x), x]
        for conv in self.convs:
            if self.use_id is None:
                x = conv(x, edge_index, z_emb)
            else:
                if self.edge_nest:
                    x = conv(x, edge_index, data.node_id) + conv(x, edge_index, data.node_id + 1)
                else:
                    x = conv(x, edge_index, data.node_id)
            xs += [x]

        if self.graph_pred:
            # x = global_add_pool(x, data.batch)
            x = global_mean_pool(torch.cat(xs, dim = 1), batch)
        else:
            x = torch.cat(xs, dim = 1)
            #x = x
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x#, []

# Provably Powerful Graph Networks
def diag_offdiag_meanpool(input, level='graph'):
    N = input.shape[-1]
    if level == 'graph':
        mean_diag = torch.mean(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)  # BxS
        mean_offdiag = (torch.sum(input, dim=[-1, -2]) - mean_diag * N) / (N * N - N)
    else:
        mean_diag = torch.diagonal(input, dim1=-2, dim2=-1)
        mean_offdiag = (torch.sum(input, dim=-1) + torch.sum(input, dim=-2) - 2 * mean_diag) # / (2 * N - 2)
    return torch.cat((mean_diag, mean_offdiag), dim=1)  # output Bx2S

class PPGN_eff(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, dataset, emb_dim=64, use_embedding=False, use_spd=False, y_ndim=1,
                 **kwargs):
        super(PPGN_eff, self).__init__()

        self.use_embedding = use_embedding
        self.use_spd = use_spd
        self.y_ndim = y_ndim
        
        initial_dim = 2
        if "h" in kwargs:
            if kwargs["h"] is not None:
                initial_dim += emb_dim
        
        if self.use_spd:
            initial_dim += 1

        # ppgn modules
        num_blocks = 2
        num_rb_layers = 4
        num_fc_layers = 2
        
        self.z_embedding = Sequential(Linear(1800, emb_dim),
                                      BN(emb_dim),
                                      ReLU(),
                                      Linear(emb_dim, emb_dim),
                                      BN(emb_dim),
                                      ReLU()
                                      )
        
        self.ppgn_rb = torch.nn.ModuleList()
        self.ppgn_rb.append(RegularBlock(num_blocks, initial_dim, emb_dim))
        for i in range(num_rb_layers - 1):
            self.ppgn_rb.append(RegularBlock(num_blocks, emb_dim, emb_dim))

        self.ppgn_fc = torch.nn.ModuleList()
        self.ppgn_fc.append(FullyConnected(emb_dim * 2, emb_dim))
        for i in range(num_fc_layers - 2):
            self.ppgn_fc.append(FullyConnected(emb_dim, emb_dim))
        self.ppgn_fc.append(FullyConnected(emb_dim, 1, activation_fn=None))

    def forward(self, data):
        # prepare dense data
        device = data.edge_index.device
        if data.edge_pos is not None:
            edge_embedding = self.z_embedding(data.edge_pos.to(torch.float).to(device))
        else:
            edge_embedding = None
        if type(data.num_nodes) is int:
            node_embedding = torch.zeros(data.num_nodes).to(device)
        else:
            node_embedding = torch.zeros(data.num_nodes.sum()).to(device)
        edge_adj = torch.ones(data.edge_index.shape[1], 1).to(device)
        if edge_embedding is not None:
            edge_data = torch.cat([edge_adj, edge_embedding], 1)
        else:
            edge_data = edge_adj
        dense_edge_data = to_dense_adj(
            data.edge_index, data.batch, edge_data
        )  # |graphs| * max_nodes * max_nodes * edge_data_dim
        if dense_edge_data.ndim <= 3:
            dense_edge_data = torch.unsqueeze(dense_edge_data, -1)
        dense_node_data, mask = to_dense_batch(node_embedding, data.batch) # |graphs| * max_nodes * d
        dense_node_data = torch.unsqueeze(dense_node_data, -1)
        shape = dense_node_data.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_data = torch.empty(*shape).to(device)

        if self.use_spd:
            dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(device)

        for g in range(shape[0]):
            for i in range(shape[-1]):
                diag_node_data[g, :, :, i] = torch.diag(dense_node_data[g, :, i])

        if self.use_spd:
            z = torch.cat([dense_dist_mat, dense_edge_data, diag_node_data], -1)
        else:
            z = torch.cat([dense_edge_data, diag_node_data], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        for rb in self.ppgn_rb:
            z = rb(z)

        # z = diag_offdiag_maxpool(z)
        if self.y_ndim == 1:
            z = diag_offdiag_meanpool(z, level='graph')
        else:
            z = diag_offdiag_meanpool(z, level='node')
        # reshape
        if z.ndim == 3:
            z = torch.transpose(z, -1, -2)
        for fc in self.ppgn_fc:
            z = fc(z)
        if z.ndim == 3:
            # from dense to sparse
            z = z.view(-1, 1)
            z = z[mask.view(-1)]
        torch.cuda.empty_cache()
        return z





# General settings.
parser = argparse.ArgumentParser(description='NestedGNN for counting experiments.')
parser.add_argument('--model', default="NestedGIN_eff", type=str, help='NestedGIN_eff/PPGN_eff')
parser.add_argument('--target', default=3, type=int) # 0 for detection of tri-cycle, 3,4,...,8 for counting of cycles
parser.add_argument('--ab', action='store_true', default=False)

# Base GNN settings.
parser.add_argument('--layers', type=int, default=5)

# Nested GNN settings
parser.add_argument('--h', type=int, default=3, help='hop of enclosing subgraph;\
                    if None, will not use NestedGNN')
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='hop',
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".')

# Training settings.
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=10)

# Other settings.
parser.add_argument('--normalize_x', action='store_true', default=False,
                    help='if True, normalize non-binary node features')
parser.add_argument('--not_normalize_dist', action='store_true', default=False,
                    help='do not normalize node distance by max distance of a molecule')
parser.add_argument('--RNI', action='store_true', default=False,
                    help='use node randomly initialized node features in [-1, 1]')
parser.add_argument('--use_relative_pos', action='store_true', default=False,
                    help='use relative node position (3D) as continuous edge features')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--dataset', default='count_cycle', help = "count_cycle/count_graphlet")
parser.add_argument('--load_model', default=None)
parser.add_argument('--eval', default=0, type=int)
parser.add_argument('--train_only', default=0, type=int)
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'results/' + args.dataset + '_' + args.save_appendix
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
# Backup python files.
copy('run_graphcount.py', args.res_dir)
copy('utils_edge_efficient.py', args.res_dir)
copy('kernel/gin.py', args.res_dir)
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


target = int(args.target)
print('---- Target: {} ----'.format(target))

path = 'data/Count'

pre_transform = None
if args.h is not None:
    if type(args.h) == int:
        path += '/ngnn_h' + str(args.h)
    elif type(args.h) == list:
        path += '/ngnn_h' + ''.join(str(h) for h in args.h)
    path += '_' + args.node_label
    if args.max_nodes_per_hop is not None:
        path += '_mnph{}'.format(args.max_nodes_per_hop)
    def pre_transform(g):
        return create_subgraphs(g, args.h,
                                max_nodes_per_hop=args.max_nodes_per_hop,
                                node_label=args.node_label,
                                use_rd=True, self_loop = True)


pre_filter = None
if args.model == "NestedGIN_eff":
    my_pre_transform = pre_transform
else:
    my_pre_transform = None



# counting benchmark
dataname = args.dataset
processed_name = dataname
if args.h is not None:
    processed_name = processed_name + "_h" + str(args.h)

train_dataset = dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                        pre_transform=my_pre_transform, split='train')
val_dataset = dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                        pre_transform=my_pre_transform, split='val')
test_dataset = dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                        pre_transform=my_pre_transform, split='test')


# ablation study for I2GNN
if args.ab:
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.data.z[:, 2:] = torch.zeros_like(dataset.data.z[:, 2:])



# normalize target
y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
mean = y_train_val.mean(dim=0)
std = y_train_val.std(dim=0)

train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_dataset.data.y = (test_dataset.data.y - mean) / std

print('Mean = %.3f, Std = %.3f' % (mean[args.target], std[args.target]))



test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

kwargs = {
    'num_layers': args.layers,
    'edge_attr_dim': 1,
    'target': args.target,
    'y_ndim': 2,
}

if args.model == "NestedGIN_eff":
    model = NestedGIN_eff(train_dataset, args.layers, 256, use_rd = True, graph_pred = False, dropout = 0, edge_nest = True, use_cycle = True)
elif args.model == "PPGN_eff":
    model = PPGN_eff(train_dataset, emb_dim=196, y_ndim = 2, h = args.h)
else:
    print("Model not implemented")
    raise NotImplementedError

if args.load_model != None:
    cpt = torch.load(args.load_model)
    model.load_state_dict(cpt)
print('Using ' + model.__class__.__name__ + ' model')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',factor=args.lr_decay_factor, patience=args.patience, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for t, data in enumerate(train_loader):
        if type(data) == dict:
            data = {key: data_.to(device) for key, data_ in data.items()}
            num_graphs = data[args.h[0]].num_graphs
        else:
            data = data.to(device)
            num_graphs = data.num_graphs
        optimizer.zero_grad()
        y = data.y
        y = y.view([y.size(0), 1])
        #print((data.edge_pos[:, -1300:][-len(data.y):, 259] - 2 * y.view(-1)).sum())
        #print(y.view(-1))
        #data.x = data.node_pos[-len(data.y):, :]
        Loss = torch.nn.L1Loss()
        loss = Loss(model(data), y)
        #loss = Loss(model(data.x), y)
        loss.backward()
        loss_all += loss * y.size(0)
        optimizer.step()
    return loss_all / train_dataset.data.y.size(0)


def test(loader):
    model.eval()
    y_true = None; y_pred = None
    error = 0; num = 0
    with torch.no_grad():
        model.eval()
        for data in loader:
            if type(data) == dict:
                data = {key: data_.to(device) for key, data_ in data.items()}
            else:
                data = data.to(device)
            y = data.y
            #data.x = data.node_pos[-len(data.y):, :]
            y_hat = model(data)[:, 0]
            #y_hat = model(data.x)[:, 0]

            error += torch.sum(torch.abs(y_hat - y))
            num += y.size(0)
    return error / num * (std[args.target])



def visualize(loader):
    model.eval()
    with torch.no_grad():
        model.eval()
        error = 0
        num = 0
        error_dict = {}
        for data in loader:
            if type(data) == dict:
                data = {key: data_.to(device) for key, data_ in data.items()}
            else:
                data = data.to(device)
            ys = (data.y * std[args.target] + mean[args.target]).int()
            y_hat = model(data)[:, 0] * std[args.target] + mean[args.target]
            for i, y in enumerate(ys):
                y = y.item()
                if y in error_dict.keys():
                    error_dict[y].append(y_hat[i].item())
                else:
                    error_dict[y] = [y_hat[i].item()]
            error += torch.sum(torch.abs(y_hat - ys))
            num += ys.size(0)
        print('Average MAE on test set: %.5f' % (error / num))

        # analysis
        nrings = []
        maes = []
        sigmas = []
        num_samples = []
        keys = list(error_dict.keys())
        keys.sort()
        for key in keys:
            pred = np.array(error_dict[key])
            mae = np.mean(np.abs(pred - key))
            sigma = np.std(np.abs(pred - key))
            nrings.append(key)
            maes.append(mae)
            sigmas.append(sigma)
            num_samples.append(pred.shape[0])
            print('graphs with %d %d-cycles: total %d, MAE = %.5f +- %.5f' % (key, args.target+3, pred.shape[0], mae, sigma))

        # plot
        maes = np.array(maes)
        sigmas = np.array(sigmas)
        import matplotlib.pyplot as plt
        plt.plot(np.array(keys), maes)
        plt.xlabel('Graphs with # 5-cycles')
        plt.ylabel('Counting MAE')
        plt.show()
        # np.save('./cpt/gnnak_random_node_2.npy', maes)
        np.save('./cpt/idgnn_random_node_2_std.npy', sigmas)

def loop(start=1, best_val_error=None):
    pbar = tqdm(range(start, args.epochs+start))
    count = 0
    for epoch in pbar:
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)
        count += 1
        if best_val_error is None:
            best_val_error = val_error
        if val_error <= best_val_error or count == 10:
            test_error = test(test_loader)
            best_val_error = val_error
            count = 0
            log = (
                    'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
                    'Test MAE: {:.7f}, Test MAE norm: {:.7f}'
            ).format(
                epoch, lr, loss, val_error,
                test_error,
                test_error / (std[target]).cuda(),
            )
            print('\n'+log+'\n')
            with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')
    model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_name)
    start = epoch + 1
    return start, best_val_error, log


best_val_error = None
start = 1

start, best_val_error, log = loop(start, best_val_error)
print(cmd_input[:-1])
print(log)
with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
    f.write(log + '\n')

def print_or_log(input_data, log=False, log_file_path="Debug.txt"):
    if not log:  # If not logging, we should just print
        print(input_data)
    else:  # Logging
        log_file = open(log_file_path, "a+")
        log_file.write(str(input_data) + "\r\n")
        log_file.close()  # Keep the file available throughout execution

'''
class NestedGIN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(NestedGIN, self).__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(train_dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
            ),
            train_eps=False,
            edge_dim=hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                    ),
                    train_eps=False,
                    edge_dim=hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 1)
        self.z_embedding = Sequential(Linear(900, hidden),
                                      BN(hidden),
                                      ReLU(),
                                      Linear(hidden, hidden),
                                      BN(hidden),
                                      ReLU()
                                      )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.z_embedding.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        edge_pos = data.edge_pos.float()
        z_emb = self.z_embedding(edge_pos)
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index, z_emb)
        for conv in self.convs:
            x = conv(x, edge_index, z_emb)

        # x = global_add_pool(x, data.node_to_subgraph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.batch)

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x#F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__
'''
