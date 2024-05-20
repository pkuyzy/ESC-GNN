import torch
import torch_geometric
from torch_geometric.data import Data
import pdb
import numpy as np
from torch_geometric.utils import to_dense_batch

# This is a copy from torch_geometric/data/batch.py
# which is modified to support batch asignment in subgraph level

class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    #if key == "edge_pos":
                    #    item = (item.to_dense() + cumsum[key]).to_sparse()
                    #else:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    #if key == "edge_pos":
                    #    size = item.size(data.__cat_dim__(key, data[key].to_dense()))
                    #else:
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'pos_batch':
                    cumsum[key] = cumsum[key] + data['pos_batch'].max() + 1
                elif key == 'pos_enc' or key == 'pos_index' or key == 'edge_pos' or key == "attn_bias":
                    cumsum[key] = cumsum[key]
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    if hasattr(data, "original_num_nodes"):
                        cumsum[key] = cumsum[key] + data.original_num_nodes
                    #else:
                    #    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'original_idx':
                    if hasattr(data, "original_num_nodes"):
                        cumsum[key] = cumsum[key] + data.original_num_nodes
                    #else:
                    #    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'batch_edge':
                    if hasattr(data, "original_edge_index"):
                        cumsum[key] = cumsum[key] + data.original_edge_index.size()[1]
                    #else:
                    #    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'tree_edge_index':
                    cumsum[key] = cumsum[key] + data.num_cliques
                elif key == 'atom2clique_index':
                    cumsum[key] = cumsum[key] + torch.tensor([[data.num_atoms], [data.num_cliques]])
                elif key == 'edge_index_2':
                    cumsum[key] = cumsum[key] + data.iso_type_2.shape[0]
                elif key == 'edge_index_3':
                    cumsum[key] = cumsum[key] + data.iso_type_3.shape[0]
                elif key == 'batch_2':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'batch_3':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'assignment2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment3_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment_index_2':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.num_nodes], [data.iso_type_2.shape[0]]])
                elif key == 'assignment_index_3':
                    inc = data.iso_type_2.shape[0] if 'assignment_index_2' in data else data.num_nodes
                    cumsum[key] = cumsum[key] + torch.LongTensor([[inc], [data.iso_type_3.shape[0]]])
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            #if key in ['pos_batch', 'pos_index', 'pos_enc']:
            #    continue
            if key == "attn_bias":
                batch[key] = None
                '''
                max_node = int(np.sqrt(max(bk.size(0) for bk in batch[key])))
                tmp_bk = torch.zeros(len(batch[key]), max_node, max_node, dtype = torch.long) + 100
                for cb, bk in enumerate(batch[key]):
                    num_node = int(np.sqrt(bk.size(0)))
                    tmp_bk[cb, :num_node, :num_node] = bk.reshape(num_node, num_node)
                batch[key] = tmp_bk
                '''
            else:
                item = batch[key][0]
                if torch.is_tensor(item):
                    batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
                elif isinstance(item, int) or isinstance(item, float):
                    batch[key] = torch.tensor(batch[key])

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        if 'assignment_index_2' in keys:
            cumsum['assignment_index_2'] = torch.LongTensor([[0], [0]])
        if 'assignment_index_3' in keys:
            cumsum['assignment_index_3'] = torch.LongTensor([[0], [0]])
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'tree_edge_index':
                    cumsum[key] = cumsum[key] + data.num_cliques
                elif key == 'atom2clique_index':
                    cumsum[key] = cumsum[key] + torch.tensor([[data.num_atoms], [data.num_cliques]])
                elif key == 'edge_index_2':
                    cumsum[key] = cumsum[key] + data.iso_type_2.shape[0]
                elif key == 'edge_index_3':
                    cumsum[key] = cumsum[key] + data.iso_type_3.shape[0]
                elif key == 'batch_2':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'batch_3':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'assignment2_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment3_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'assignment_index_2':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.num_nodes], [data.iso_type_2.shape[0]]])
                elif key == 'assignment_index_3':
                    cumsum[key] = cumsum[key] + torch.LongTensor([[data.iso_type_2.shape[0]], [data.iso_type_3.shape[0]]])
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
