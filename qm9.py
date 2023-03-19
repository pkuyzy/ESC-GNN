import os
import os.path as osp
import shutil
from itertools import repeat

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from tqdm import tqdm

try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None
#rdkit = None # always use processed version

HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 16 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """  # noqa: E501

    raw_url = ('https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'http://www.roemisch-drei.de/qm9.zip'

    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, skip_collate=False, one_hot_atom=True):
        self.skip_collate = skip_collate
        self.one_hot_atom = one_hot_atom
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        if self.skip_collate:
            self.data = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if rdkit is None:
            return 'qm9.pt'
        else:
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if rdkit is None:
            file_path = download_url(self.processed_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
        else:
            if os.path.isfile('data/qm9.zip'):
                extract_zip('data/qm9.zip', self.raw_dir)
            else:
                file_path = download_url(self.raw_url, self.raw_dir)
                extract_zip(file_path, self.raw_dir)
                os.unlink(file_path)

            if os.path.isfile('data/uncharacterized.txt'):
                shutil.copy('data/uncharacterized.txt', self.raw_dir)
            else:
                file_path = download_url(self.raw_url2, self.raw_dir)
                os.rename(
                    osp.join(self.raw_dir, '3195404'),
                    osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])

            # temporally disable skip_collate to read preprocessed data
            skip_collate = self.skip_collate
            self.skip_collate = False  
            data_list = [self.get(i) for i in range(len(self))]
            self.skip_collate = skip_collate

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                new_data_list = []
                for data in tqdm(data_list, ncols=70):
                    new_data_list.append(self.pre_transform(data))
                del data_list
                data_list = new_data_list

            if self.skip_collate:
                torch.save(data_list, self.processed_paths[0])
            else:
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
        assert len(skip) == 3054

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        data_list = []
        for i, mol in tqdm(enumerate(suppl), ncols=70):
            if mol is None:
                continue
            if i in skip:
                continue

            text = suppl.GetItemText(i)
            N = mol.GetNumAtoms()

            pos = text.split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            acceptor = []
            donor = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                donor.append(0)
                acceptor.append(0)
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x2 = torch.tensor([
                atomic_number, acceptor, donor, aromatic, sp, sp2, sp3, num_hs
            ], dtype=torch.float).t().contiguous()
            if self.one_hot_atom:
                x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
                x = torch.cat([x1.to(torch.float), x2], dim=-1)
            else:
                node_type = torch.tensor(type_idx)
                x = x2
                

            row, col, bond_idx = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = F.one_hot(
                torch.tensor(bond_idx),
                num_classes=len(self.bonds)).to(torch.float)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            data = Data(x=x, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name)
            if not self.one_hot_atom:
                data.node_type = node_type

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    # overwrite original len() and get() in InMemoryDataset to handle skip_collate
    def len(self):
        if self.skip_collate:
            return len(self.data)
        for item in self.slices.values():
            return len(item) - 1
        return 0

    def get(self, idx):
        if self.skip_collate:
            return self.data[idx]

        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data
