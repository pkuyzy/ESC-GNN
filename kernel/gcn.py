import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
#from kernel.gcn_conv import GCNConv
import pdb
from kernel.idgnn import GCNIDConvLayer


class NestedGCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, use_z=False, use_rd=False, use_cycle = False, graph_pred = True, use_id = None, dropout = 0.2, multi_layer = False, edge_nest = False):
        super(NestedGCN, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.graph_pred = graph_pred # delete the final graph-level pooling
        self.use_cycle = use_cycle # to mark whther predicting the cycle or not
        self.use_id = use_id
        self.dropout = dropout # dropout 0.1 for multilayer, 0.2 for no multi
        self.multi_layer = multi_layer # to use multi layer supervision or not
        self.edge_nest = edge_nest # denote whether using the edge-level nested information
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
        input_dim = dataset.num_features
        if self.use_z or self.use_rd:
            input_dim += 8

        if use_id is None:
            self.conv1 = GCNConv(input_dim, hidden)
        else:
            self.conv1 = GCNIDConvLayer(input_dim, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if use_id is None:
                self.convs.append(GCNConv(hidden, hidden))
            else:
                self.convs.append(GCNIDConvLayer(hidden, hidden))
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1))

        # for multilayer training
        self.multi_lin = torch.nn.ModuleList()
        self.multi_bn = torch.nn.ModuleList()
        self.multi_lin2 = torch.nn.ModuleList()
        for i in range(1, num_layers):
            self.multi_lin.append(torch.nn.Linear(i * hidden, hidden))
            self.multi_bn.append(torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1))
            self.multi_lin2.append(torch.nn.Linear(hidden, 2*i-1))

        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        #self.lin1 = torch.nn.Linear(hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not use_cycle:
            # original code
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 5)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
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
            x = self.bn1(self.conv1(x, edge_index))
        else:
            x = self.bn1(self.conv1(x, edge_index, data.node_id))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        xs = [x]

        #multilayer training
        ys = []
        for bn, conv, ml, mb, ml2 in zip(self.bns, self.convs, self.multi_lin, self.multi_bn, self.multi_lin2):

            if self.multi_layer:
                if not self.edge_nest:
                    ys += [ml2(F.relu(F.dropout(mb(ml(torch.cat(xs, dim=1)[data.node_id])), p = self.dropout, training=self.training)))]
                else:
                    tmp_x = global_add_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
                    tmp_x = global_add_pool(tmp_x, data.original_edge_index[0]) + global_add_pool(tmp_x, data.original_edge_index[1])
                    ys + [ml2(F.relu(F.dropout(mb(ml(tmp_x)), p = self.dropout, training=self.training)))]

            if self.use_id is None:
                x = bn(conv(x, edge_index))
            else:
                if self.edge_nest:
                    #x = bn(conv(x, edge_index, torch.cat((data.node_id, data.node_id + 1))))
                    x = bn(conv(x, edge_index, data.node_id) + conv(x, edge_index, data.node_id + 1))
                else:
                    x = bn(conv(x, edge_index, data.node_id))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(x)
            xs += [x]

        #if self.use_id != 'ID':
        #    x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        #else:
        #    x = torch.cat(xs, dim=1)[data.node_id]

        if not self.edge_nest:
            x = global_add_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
            #x = torch.cat(xs, dim=1)[data.node_id]
            #x = global_mean_pool(x, data.node_to_subgraph)
        else:
            if not self.graph_pred:
                # for node-level tasks, such as cycle regression, first obatin the edge information, the pooling to get the node information
                x = global_mean_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
                x = global_add_pool(x, data.original_edge_index[0]) + global_add_pool(x, data.original_edge_index[1])
                #x = global_add_pool(torch.cat(xs, dim=1)[data.node_id], data.original_edge_index[0]) + global_add_pool(torch.cat(xs, dim=1)[data.node_id + 1], data.original_edge_index[1])

                # in case some out-of-graph nodes which are out of the max index in edge
                if x.size()[0] < data.original_num_nodes.sum():
                    x = torch.cat(
                        (x, torch.zeros(data.original_num_nodes.sum() - x.size()[0], x.size()[1]).to(x.device)), dim=0)
            else:
                x = global_add_pool(torch.cat(xs, dim=1), data.node_to_subgraph)
        if self.graph_pred:
            x = global_mean_pool(x, data.subgraph_to_graph)
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x, ys

    def __repr__(self):
        return self.__class__.__name__


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, *args, **kwargs):
        super(GCN, self).__init__()
        if "graph_pred" in kwargs:
            self.graph_pred = kwargs["graph_pred"] # delete the final graph-level pooling
        else:
            self.graph_pred = True
        if "use_cycle" in kwargs:
            self.use_cycle = kwargs["use_cycle"] # to mark whther predicting the cycle or not
        else:
            self.use_cycle = False
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1))
        if self.graph_pred:
            self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.bn_lin1 = torch.nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        if not self.use_cycle:
            # original code
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            self.lin2 = Linear(hidden, 5)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x.float(), edge_index)))
        xs = [x]
        for bn, conv in zip(self.bns, self.convs):
            x = F.relu(bn(conv(x, edge_index)))
            xs += [x]
        if self.graph_pred:
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = torch.cat(xs, dim = 1)
        x = F.relu(self.bn_lin1(self.lin1(x)))
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.lin2(x)
        if not self.use_cycle:
            return F.log_softmax(x, dim=-1)
        else:
            return x, []

    def __repr__(self):
        return self.__class__.__name__
