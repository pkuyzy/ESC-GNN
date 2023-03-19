import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import os.path as osp
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, BatchNorm1d as BN
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, GINEConv, global_mean_pool
import torch_geometric.transforms as T
#from PlanarSATPairsDataset import PlanarSATPairsDataset
from torch_geometric.datasets import GNNBenchmarkDataset
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
#from utils import create_subgraphs
from utils_edge_efficient import create_subgraphs
import pdb
from kernel.train_eval import k_fold



parser = argparse.ArgumentParser(description='Nested GNN for CSL datasets')
parser.add_argument('--model', type=str, default='GIN')    # Base GNN used, GIN or GCN
parser.add_argument('--h', type=int, default=4,
                    help='largest height of rooted subgraphs to simulate')
parser.add_argument('--layers', type=int, default=5)   # Number of GNN layers
parser.add_argument('--width', type=int, default=128)    # Dimensionality of GNN embeddings
parser.add_argument('--epochs', type=int, default=500)    # Number of training epochs
parser.add_argument('--dataset', type=str, default='CSL')    # Dataset being used
parser.add_argument('--learnRate', type=float, default=1E-3)   # Learning Rate
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
    MODEL = MODEL+"lr"+str(LEARNING_RATE)+"-"

BATCH = 64
MODULO = 4
MOD_THRESH = 1

path = 'data/' + DATASET
pre_transform = None
if args.h is not None:
    def pre_transform(g):
        return create_subgraphs(g, args.h, node_label='hop', use_rd=True,
                                subgraph_pretransform=None, self_loop = True)
#shutil.rmtree(path + '/processed')
 
dataset = GNNBenchmarkDataset(root=path, name = args.dataset,
                                #pre_transform=T.Compose([MyPreTransform(), pre_transform]),
                                pre_transform=pre_transform
                              )


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
                Linear(1, hidden),
                #ReLU(),
                ELU(),
                Linear(hidden, hidden),
                #ReLU(),
                ELU(),
            ),
            train_eps=False,
            edge_dim = hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden, hidden),
                        #ReLU(),
                        ELU(),
                        Linear(hidden, hidden),
                        #ReLU(),
                        ELU(),
                    ),
                    train_eps=False,
                    edge_dim = hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 10)
        self.z_initial = torch.nn.Embedding(1800, hidden)
        self.z_embedding = Sequential(
                                      torch.nn.BatchNorm1d(hidden),
                                      ELU(),
                                      Linear(hidden, hidden),
                                      torch.nn.BatchNorm1d(hidden),
                                      ELU()
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
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index, z_emb)
        for conv in self.convs:
            x = conv(x, edge_index, z_emb)

        #x = global_add_pool(x, data.node_to_subgraph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        x = global_add_pool(x, data.batch)
        #x = global_mean_pool(x, data.batch)

        #x = F.relu(self.lin1(x))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x#F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'GIN':
    model = NestedGIN(args.layers, args.width).to(device)
else:
    raise NotImplementedError('model type not supported')

'''
def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)
'''

def train(epoch, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        num_graphs = data.num_graphs
        data = data.to(device)
        y = data.y
        out = model(data).squeeze()
        loss = F.cross_entropy(out, y)
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
    return total_loss / len(loader.dataset)
'''
def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)
'''
@torch.no_grad()
def val(loader):
    model.eval()
    total_loss = 0
    for data in loader:
        num_graphs = data.num_graphs
        data = data.to(device)
        y = data.y
        out = model(data).squeeze()
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * num_graphs
    return total_loss / len(loader.dataset)

'''
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        nb_trials = 1   # Support majority vote, but single trial is default
        successful_trials = torch.zeros_like(data.y)
        for i in range(nb_trials):  # Majority Vote
            pred = model(data).max(1)[1]
            successful_trials += pred.eq(data.y)
        successful_trials = successful_trials > (nb_trials // 2)
        correct += successful_trials.sum().item()
    return correct / len(loader.dataset)
'''
@torch.no_grad()
def test(loader):
    model.eval()
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y = data.y
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    '''
    print(y_preds[y_preds == y_trues])
    print(y_trues[y_preds == y_trues])
    print(y_preds[~(y_preds == y_trues)])
    print(y_trues[~(y_preds == y_trues)])
    '''
    return (y_preds == y_trues).float().mean()

acc = []
tr_acc = []
#SPLITS = 2
SPLITS = 10
tr_accuracies = np.zeros((EPOCHS, SPLITS))
tst_accuracies = np.zeros((EPOCHS, SPLITS))
tst_exp_accuracies = np.zeros((EPOCHS, SPLITS))
tst_lrn_accuracies = np.zeros((EPOCHS, SPLITS))

#for i in range(SPLITS):
for i, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, SPLITS))):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=LEARNING_RATE)
    '''
    n = len(dataset) // SPLITS
    val_mask = torch.zeros(len(dataset), dtype=torch.bool)

    val_mask[i * n:(i + 1) * n] = 1 # Now set the masks

    # Now load the datasets
    val_dataset = dataset[val_mask]
    train_dataset = dataset[~val_mask]

    n = len(train_dataset) // SPLITS
    test_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
    test_mask[i * n:(i + 1) * n] = 1
    test_dataset = train_dataset[test_mask]
    train_dataset = train_dataset[~test_mask]
    '''
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    val_dataset = dataset[val_idx]

    val_loader = DataLoader(val_dataset, batch_size=BATCH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    print_or_log('---------------- Split {} ----------------'.format(i),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    best_val_loss, test_acc = 100, 0
    for epoch in tqdm(range(EPOCHS)):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            best_val_loss = val_loss
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_loss = val(test_loader)
        test_acc = test(test_loader)
        tr_accuracies[epoch, i] = train_acc
        tst_accuracies[epoch, i] = test_acc
        print_or_log('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
              'Val Loss: {:.7f}, Val Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, Train Acc: {:.7f}'.format(
                  epoch+1, lr, train_loss, val_loss, val_acc, test_loss, test_acc, train_acc),log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
    acc.append(test_acc)
    tr_acc.append(train_acc)

acc = torch.tensor(acc)
tr_acc = torch.tensor(tr_acc)
print_or_log('---------------- Final Result ----------------',
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Tr Mean: {:7f}, Std: {:7f}'.format(tr_acc.mean(), tr_acc.std()),
             log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")


'''
print_or_log('Average Acros Splits', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
print_or_log('Training Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tr_accuracies = np.mean(tr_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tr_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_accuracies = np.mean(tst_accuracies, axis=1)
st_d_tst_accuracies = np.std(tst_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_accuracies[epoch])+"/"+str(st_d_tst_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Exp Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_e_accuracies = np.mean(tst_exp_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_e_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")

print_or_log('Testing Lrn Acc:', log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
mean_tst_l_accuracies = np.mean(tst_lrn_accuracies, axis=1)
for epoch in range(EPOCHS):
    print_or_log('Epoch '+str(epoch+1)+':'+str(mean_tst_l_accuracies[epoch]),
                 log_file_path="log"+MODEL+DATASET+","+str(LAYERS)+","+str(WIDTH)+".txt")
'''
