import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, GINEConv
import torch_geometric.transforms as T
#from PlanarSATPairsDataset import PlanarSATPairsDataset
from SRDataset import SRDataset
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
# from utils import create_subgraphs
from utils_edge_efficient import create_subgraphs
import pdb
from kernel.gin import NestedGIN_eff as NestedGIN


parser = argparse.ArgumentParser(description='Nested GNN for EXP/CEXP datasets')
parser.add_argument('--model', type=str, default='GIN')  # Base GNN used, GIN or GCN
parser.add_argument('--h', type=int, default=3,
                    help='largest height of rooted subgraphs to simulate')
parser.add_argument('--layers', type=int, default=8)  # Number of GNN layers
parser.add_argument('--width', type=int, default=64)  # Dimensionality of GNN embeddings
parser.add_argument('--epochs', type=int, default=500)  # Number of training epochs
parser.add_argument('--dataset', type=str, default='EXP')  # Dataset being used
parser.add_argument('--learnRate', type=float, default=0.001)  # Learning Rate
args = parser.parse_args()


def print_or_log(input_data, log=False, log_file_path="Debug.txt"):
    if not log:  # If not logging, we should just print
        print(input_data)
    else:  # Logging
        log_file = open(log_file_path, "a+")
        log_file.write(str(input_data) + "\r\n")
        log_file.close()  # Keep the file available throughout execution


class MyFilter(object):
    def __call__(self, data):
        return True  # No Filtering


class MyPreTransform(object):
    def __call__(self, data):
        data.x = F.one_hot(data.x[:, 0], num_classes=2).to(torch.float)  # Convert node labels to one-hot
        return data


# Command Line Arguments
DATASET = args.dataset
LAYERS = args.layers
EPOCHS = args.epochs
WIDTH = args.width
LEARNING_RATE = args.learnRate

MODEL = f"Nested{args.model}-"

if LEARNING_RATE != 0.001:
    MODEL = MODEL + "lr" + str(LEARNING_RATE) + "-"

BATCH = 20
MODULO = 4
MOD_THRESH = 1

path = 'data/' + DATASET
pre_transform = None
if args.h is not None:
    def pre_transform(g):
        return create_subgraphs(g, args.h, node_label='hop', use_rd=False,
                                subgraph_pretransform=None, self_loop=True)
# shutil.rmtree(path + '/processed')

dataset = SRDataset(root="data/sr25/", pre_transform = pre_transform)#pre_transform=T.Compose([MyPreTransform(), pre_transform]))

'''
class NestedGIN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(NestedGIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
            ),
            train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                    ),
                    train_eps=False))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__
'''


class NestedGIN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(NestedGIN, self).__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(dataset.num_features, hidden),
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
        self.lin2 = Linear(hidden, hidden)
        self.z_initial = torch.nn.Embedding(1800, hidden)
        self.z_embedding = Sequential(
                                      torch.nn.BatchNorm1d(hidden),
                                      ReLU(),
                                      Linear(hidden, hidden),
                                      torch.nn.BatchNorm1d(hidden),
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
        if hasattr(data, 'edge_pos'):
            # original, slow version
            edge_pos = data.edge_pos.float()
            z_emb = torch.mm(edge_pos, self.z_initial.weight)
        else:
            # new, fast version
            z_emb = global_add_pool(
                torch.mul(self.z_initial.weight[data.pos_index], data.pos_enc.view(-1, 1)),
                data.pos_batch)
        z_emb = self.z_embedding(z_emb)
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index, z_emb)
        for conv in self.convs:
            x = conv(x, edge_index, z_emb)

        # x = global_add_pool(x, data.node_to_subgraph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        x = global_add_pool(x, data.batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'GIN':
    model = NestedGIN(args.layers, args.width).to(device)
else:
    raise NotImplementedError('model type not supported')

total_wrong = 0
total_num = 0
if True:
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=LEARNING_RATE)


    dataloader = DataLoader(dataset, batch_size=BATCH)
    model.eval()
    with torch.no_grad():
        Pred = []
        for cd, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data)
            Pred.append(pred)
        Pred = torch.cat(Pred, dim=0)
        mm = torch.pdist(Pred, p=2)
        wrong = (mm < 1e-2).sum().item()
        total_wrong += wrong
        total_num += mm.shape[0]
        test_score = 1 - (wrong / mm.shape[0])

print_or_log('---------------- Final Result ----------------',
             log_file_path="log" + MODEL + DATASET + "," + str(LAYERS) + "," + str(WIDTH) + ".txt")
print_or_log('Acc: {}'.format(test_score),
             log_file_path="log" + MODEL + DATASET + "," + str(LAYERS) + "," + str(WIDTH) + ".txt")
