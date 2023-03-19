import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool
from kernel.idgnn import GINIDConvLayer
import time


class NestedGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False, use_cycle=False, graph_pred=True,
                 use_id=None, dropout=0.2, multi_layer=False, edge_nest=False):
        super(NestedGIN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.graph_pred = graph_pred  # delete the final graph-level pooling
        self.use_cycle = use_cycle  # to mark whther predicting the cycle or not
        self.use_id = use_id
        self.dropout = dropout  # dropout 0.1 for multilayer, 0.2 for no multi
        self.multi_layer = multi_layer  # to use multi layer supervision or not
        self.edge_nest = edge_nest  # denote whether using the edge-level nested information
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        if use_id is None:
            # self.conv1 = GCNConv(input_dim, hidden)
            self.conv1 = GINConv(
                Sequential(
                    Linear(input_dim, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                ),
                train_eps=True)
        else:
            self.conv1 = GINIDConvLayer(Sequential(
                Linear(input_dim, hidden),
                BN(hidden),
                Dropout(dropout),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                Dropout(dropout),
                ReLU(),
            ), Sequential(
                Linear(input_dim, hidden),
                BN(hidden),
                Dropout(dropout),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                Dropout(dropout),
                ReLU(),
            ),
                train_eps=True)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if use_id is None:
                self.convs.append(GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        Dropout(dropout),
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        Dropout(dropout),
                        ReLU(),
                    ),
                    train_eps=True))
            else:
                self.convs.append(GINIDConvLayer(Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                ), Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    Dropout(dropout),
                    ReLU(),
                ),
                    train_eps=True))

        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not use_cycle:
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 5)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        data.to(self.lin1.weight.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # node label embedding
        z_emb = 0
        if self.use_z and 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        if self.use_rd or self.use_z:
            x = torch.cat([z_emb, x], -1)

        if self.use_id is None:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index, data.node_id)
        # x = self.conv1(x, edge_index)

        xs = [x]
        for conv in self.convs:
            if self.use_id is None:
                x = conv(x, edge_index)
            else:
                if self.edge_nest:
                    # x = bn(conv(x, edge_index, torch.cat((data.node_id, data.node_id + 1))))
                    x = conv(x, edge_index, data.node_id) + conv(x, edge_index, data.node_id + 1)
                else:
                    x = conv(x, edge_index, data.node_id)
            # x = conv(x, edge_index)
            xs += [x]

        if not self.edge_nest:
            x = global_add_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
            # x = torch.cat(xs, dim=1)[data.node_id]
            # x = global_mean_pool(x, data.node_to_subgraph)
        else:
            if not self.graph_pred:
                # for node-level tasks, such as cycle regression, first obatin the edge information, the pooling to get the node information
                x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
                x = global_add_pool(x, data.original_edge_index[0]) + global_add_pool(x, data.original_edge_index[1])
                # x = global_add_pool(torch.cat(xs, dim=1)[data.node_id], data.original_edge_index[0]) + global_add_pool(torch.cat(xs, dim=1)[data.node_id + 1], data.original_edge_index[1])

                # in case some out-of-graph nodes which are out of the max index in edge
                if x.size()[0] < data.original_num_nodes.sum():
                    x = torch.cat(
                        (x, torch.zeros(data.original_num_nodes.sum() - x.size()[0], x.size()[1]).to(x.device)), dim=0)
            else:
                x = global_add_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        # x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)

        if self.graph_pred:
            # x = global_add_pool(x, data.subgraph_to_graph)
            x = global_mean_pool(x, data.subgraph_to_graph)
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x, []

        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


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
        #self.z_embedding = Sequential(Linear(z_in, hidden),
        #                              Dropout(dropout),
        #                              BN(hidden),
        #                              ReLU(),
        #                              Linear(hidden, hidden),
        #                              Dropout(dropout),
        #                              BN(hidden),
        #                              ReLU()
        #                              )
        self.z_initial = torch.nn.Embedding(z_in, hidden)
        self.z_embedding = Sequential(Dropout(dropout),
                                      torch.nn.BatchNorm1d(hidden),
                                      ReLU(),
                                      Linear(hidden, hidden),
                                      Dropout(dropout),
                                      torch.nn.BatchNorm1d(hidden),
                                      ReLU()
                                      )
        input_dim = dataset.num_features
        #if self.use_z or self.use_rd:
        #    input_dim += 8

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
        else:
            self.conv1 = GINIDConvLayer(Sequential(
                Linear(input_dim, hidden),
                Dropout(dropout),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                Dropout(dropout),
                BN(hidden),
                ReLU(),
            ), Sequential(
                Linear(input_dim, hidden),
                Dropout(dropout),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                Dropout(dropout),
                BN(hidden),
                ReLU(),
            ),
                train_eps=True)

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
            else:
                self.convs.append(GINIDConvLayer(Sequential(
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                ), Sequential(
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    Dropout(dropout),
                    BN(hidden),
                    ReLU(),
                ),
                    train_eps=True))

        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
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
        #edge_pos = data.edge_pos.float()

        #z_emb = self.z_embedding(edge_pos)

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

        if self.use_id is None:
            x = self.conv1(x, edge_index, z_emb)
        else:
            x = self.conv1(x, edge_index, data.node_id)

        xs = [x]
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
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x#, []

class GIN0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, subconv=False):
        super(GIN0, self).__init__()
        self.subconv = subconv
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                BN(hidden),
                ReLU(),
                Linear(hidden, hidden),
                BN(hidden),
                ReLU(),
            ),
            train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        ReLU(),
                    ),
                    train_eps=False))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if True:
            if self.subconv:
                x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
                x = global_add_pool(x, data.subgraph_to_graph)
                x = F.relu(self.lin1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.lin2(x)
            else:
                x = global_add_pool(torch.cat(xs, dim=1), batch)
                x = F.relu(self.lin1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.lin2(x)
        else:  # GIN pooling in the paper
            xs = [global_add_pool(x, batch) for x in xs]
            xs = [F.dropout(self.lin2(x), p=0.5, training=self.training) for x in xs]
            x = 0
            for x_ in xs:
                x += x_

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, *args, **kwargs):
        super(GIN, self).__init__()
        if "graph_pred" in kwargs:
            self.graph_pred = kwargs["graph_pred"]  # delete the final graph-level pooling
        else:
            self.graph_pred = True
        if "use_cycle" in kwargs:
            self.use_cycle = kwargs["use_cycle"]  # to mark whther predicting the cycle or not
        else:
            self.use_cycle = False

        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not self.use_cycle:
            # original code
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 5)
        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        if self.graph_pred:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = torch.cat(xs, dim=1)
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        x = F.relu(self.bn_lin1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x, []
        # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
