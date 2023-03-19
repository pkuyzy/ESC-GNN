import torch
import random
import math
import pdb
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from scipy import linalg
from scipy.linalg import inv, eig, eigh
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_scatter import scatter_min
from batch import Batch
from collections import defaultdict
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_geometric.utils import degree


def create_subgraphs(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                     node_label='hop', use_rd=False, subgraph_pretransform=None, data_name=None, self_loop=False):
    # Given a PyG graph data, extract an h-hop rooted subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph
    # If given a list of h, will return multiple subgraphs for each node stored in
    # a dict.

    if type(h) == int:
        h = [h]
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    if type(data.num_nodes) is torch.Tensor:
        num_nodes = num_nodes.item()
    if self_loop:
        #edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr=data.edge_attr, num_nodes=num_nodes)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=data.edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    else:
        edge_attr = data.edge_attr

    new_data_multi_hop = {}
    for h_ in h:
        subgraphs = []
        for e in edge_index.T:
            nodes_0, edge_index_0, edge_mask_0, z_0 = k_hop_subgraph(
                e[0], h_, edge_index, False, num_nodes, node_label='hop',
                max_nodes_per_hop=max_nodes_per_hop
            )
            nodes_1, edge_index_1, edge_mask_1, z_1 = k_hop_subgraph(
                e[1], h_, edge_index, False, num_nodes, node_label='hop',
                max_nodes_per_hop=max_nodes_per_hop
            )
            nodes_ = [nodes_0[0], nodes_1[0]]
            nodes_ = nodes_ + [item for item in nodes_0 if item not in nodes_]
            nodes_ = nodes_ + [item for item in nodes_1 if item not in nodes_]
            edge_mask_ = torch.logical_or(edge_mask_0, edge_mask_1)
            z_ = []
            for n in nodes_:
                d0 = z_0[n] if n in z_0 else h_ + 1;
                d1 = z_1[n] if n in z_1 else h_ + 1
                z_.append([d0, d1])
            z_ = torch.tensor(z_).to(edge_index.device)
            nodes_ = torch.tensor(nodes_).to(edge_index.device)
            edge_index_ = edge_index[:, edge_mask_]
            # relabel nodes
            node_idx = edge_index[1].new_full((num_nodes,), -1)
            node_idx[nodes_] = torch.arange(nodes_.size(0), device=edge_index.device)
            edge_index_ = node_idx[edge_index_]

            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]
            data_.sub_degree = degree(edge_index_[0], num_nodes = nodes_.shape[0])
            data_.original_idx = nodes_

            if 'node_type' in data:
                data_.node_type = node_type_

            if use_rd:
                # See "Link prediction in complex networks: A survey".
                adj = to_scipy_sparse_matrix(
                    edge_index_, num_nodes=nodes_.shape[0]
                ).tocsr()
                laplacian = ssp.csgraph.laplacian(adj).toarray()
                try:
                    L_inv = linalg.pinv(laplacian)
                except:
                    laplacian += 0.01 * np.eye(*laplacian.shape)
                lxx = L_inv[0, 0]
                lyy = L_inv[list(range(len(L_inv))), list(range(len(L_inv)))]
                lxy = L_inv[0, :]
                lyx = L_inv[:, 0]
                rd_to_x = torch.FloatTensor((lxx + lyy - lxy - lyx)).unsqueeze(1)
                data_.rd = rd_to_x

            if subgraph_pretransform is not None:  # for k-gnn
                data_ = subgraph_pretransform(data_)
                if 'assignment_index_2' in data_:
                    data_.batch_2 = torch.zeros(
                        data_.iso_type_2.shape[0], dtype=torch.long
                    )
                if 'assignment_index_3' in data_:
                    data_.batch_3 = torch.zeros(
                        data_.iso_type_3.shape[0], dtype=torch.long
                    )

            subgraphs.append(data_)

        pos_encs = []
        pos_indices = []
        pos_batches = []
        cnt_batch = 0
        for sg in subgraphs:
            # encodes the distance and degree information of nodes within the subgraph
            lsg = sg.num_nodes
            pos_enc = torch.cat((F.one_hot(sg.sub_degree.long(), num_classes = 200).view(lsg, -1), F.one_hot(sg.z.long(), num_classes = 100).view(lsg, -1)), dim = -1)
            if use_rd:
                pos_enc = torch.cat((pos_enc, F.one_hot(sg.rd.long(), num_classes = 100).view(lsg, -1)), dim = -1)
            pos_enc = pos_enc.sum(dim = 0)

            # encodes the edge information within the subgraph
            # wrong version, cannot count number of edges of certain type
            #pos_enc = torch.cat((pos_enc, F.one_hot(sg.z[sg.edge_index].transpose(0, 1).reshape(-1, 4), num_classes=100).view(-1,400).sum(dim=0)))
            pos_enc = torch.cat((pos_enc,
                                 F.one_hot((sg.z[remove_self_loops(sg.edge_index)[0]].transpose(0, 1).reshape(-1, 4))@torch.tensor([216, 36, 6, 1]), num_classes = 1300).sum(dim = 0)))
            #pos_encs.append(pos_enc.unsqueeze(0))#.to_sparse())
            pos_index = torch.nonzero(pos_enc)
            pos_encs.append(pos_enc[pos_index].view(-1))
            pos_indices.append(pos_index.view(-1))
            pos_batches.append(torch.LongTensor([cnt_batch for _ in range(pos_index.size()[0])]))
            cnt_batch += 1

        if not hasattr(data, 'pos'):
            new_data =  data.__class__(data.x, edge_index, edge_attr, data.y, None, pos_enc = torch.cat(pos_encs, dim = 0), pos_index = torch.cat(pos_indices, dim = 0), pos_batch = torch.cat(pos_batches, dim = 0))
        elif not hasattr(data, 'name'):
            new_data= data.__class__(data.x, edge_index, edge_attr, data.y, None, pos_enc = torch.cat(pos_encs, dim = 0), pos_index = torch.cat(pos_indices, dim = 0), pos_batch = torch.cat(pos_batches, dim = 0))
        else:
            new_data = data.__class__(data.x, edge_index, edge_attr, data.y, pos = data.pos, pos_enc = torch.cat(pos_encs, dim = 0), pos_index = torch.cat(pos_indices, dim = 0), pos_batch = torch.cat(pos_batches, dim = 0), name = data.name, node_type = data.node_type)
    return new_data
    '''
        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        data.original_num_nodes = data.num_nodes
        new_data.num_subgraphs = len(subgraphs)
        batch_edge = []; cbe = 0
        for data_ in subgraphs:
            batch_edge += [cbe for _ in range(data_.num_edges)]
            cbe += 1
        new_data.batch_edge = torch.LongTensor(batch_edge)
        new_data.original_x = data.x

        new_data.original_edge_index = edge_index
        new_data.original_edge_attr = data.edge_attr
        new_data.original_pos = data.pos

        # rename batch, because batch will be used to store node_to_graph assignment
        new_data.node_to_subgraph = new_data.batch
        new_data.node_id = torch.LongTensor(
            [0] + [i for i in range(1, len(new_data.batch)) if new_data.batch[i] != new_data.batch[i - 1]])

        del new_data.batch
        if 'batch_2' in new_data:
            new_data.assignment2_to_subgraph = new_data.batch_2
            del new_data.batch_2
        if 'batch_3' in new_data:
            new_data.assignment3_to_subgraph = new_data.batch_3
            del new_data.batch_3

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)

        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop
    '''


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop',
                   max_nodes_per_hop=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    if node_label == 'hop':
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h + 2)
        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(tmp):
                tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
        if node_label == 'hop':
            hops.append(torch.LongTensor([h + 1] * len(new_nodes), device=row.device))
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    if node_label == 'hop':
        hop = torch.cat(hops)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label == 'hop':
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])
        z = {ss: zz for ss, zz in zip(subset.tolist(), hop.tolist())}
    elif node_label.startswith('spd') or node_label == 'drnl':
        if node_label.startswith('spd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label == 'drnl':
            # see "Link Prediction Based on Graph Neural Networks", a special
            # case of spd2
            num_spd = 2
            z = torch.zeros([subset.size(0), 1], dtype=torch.long, device=row.device)

        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]  # keep top num_spd distances
            if node_label == 'spd':
                z[i][:min(num_spd, len(dists))] = torch.tensor(dists)
            elif node_label == 'drnl':
                dist1 = dists[0]
                dist2 = dists[1] if len(dists) == 2 else 0
                if dist2 == 0:
                    dist = dist1
                else:
                    dist = dist1 * (num_hops + 1) + dist2
                z[i][0] = dist

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset.tolist(), edge_index, edge_mask, z


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        _, out_nei, _ = ssp.find(A[node, :])
        in_nei, _, _ = ssp.find(A[:, node])
        nei = set(out_nei).union(set(in_nei))
        res = res.union(nei)
    return res


class return_prob(object):
    def __init__(self, steps=50):
        self.steps = steps

    def __call__(self, data):
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        adj += ssp.identity(data.num_nodes, dtype='int', format='csr')
        rp = np.empty([data.num_nodes, self.steps])
        inv_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
        inv_deg.setdiag(1 / adj.sum(1))
        P = inv_deg * adj
        if self.steps < 5:
            Pi = P
            for i in range(self.steps):
                rp[:, i] = Pi.diagonal()
                Pi = Pi * P
        else:
            inv_sqrt_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
            inv_sqrt_deg.setdiag(1 / (np.array(adj.sum(1)) ** 0.5))
            B = inv_sqrt_deg * adj * inv_sqrt_deg
            L, U = eigh(B.todense())
            W = U * U
            Li = L
            for i in range(self.steps):
                rp[:, i] = W.dot(Li)
                Li = Li * L

        data.rp = torch.FloatTensor(rp)

        return data




