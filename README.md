# ESC-GNN
The official code for our KDD 2024 paper "An Efficient Subgraph GNN with Provable Substructure Counting Power" ([paper](https://arxiv.org/pdf/2303.10576))

The requirement is available at [requirements.txt](requirements.txt). Key packages include torch, torch_geometric (and the related packages, including torch_scatter, torch_sparse, torch_cluster, and torch_spline_conv), networkx, and rdkit.

For installing torch, please refer to: https://pytorch.org/get-started/previous-versions/

For installing torch_geometric, please refer to: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html



Counting cycles:

```
python run_graphcount.py --batch_size 128 --target 0 --model NestedGIN_eff --h 3 --lr 1e-2
python run_graphcount.py --batch_size 128 --target 1 --model NestedGIN_eff --h 3 --lr 1e-2 --layers 5
python run_graphcount.py --batch_size 128 --target 2 --model NestedGIN_eff --h 2 --lr 5e-3
python run_graphcount.py --batch_size 128 --target 3 --model NestedGIN_eff --h 3 --lr 1e-2 --layers 5
```

Counting graphlets:

```
python run_graphcount.py --batch_size 128 --target 0 --model NestedGIN_eff --h 1 --lr 8e-3 --dataset count_graphlet
python run_graphcount.py --batch_size 256 --target 1 --model NestedGIN_eff --h 4 --lr 4e-3 --dataset count_graphlet
python run_graphcount.py --batch_size 521 --target 2 --model NestedGIN_eff --h 1 --lr 4e-3 --dataset count_graphlet
python run_graphcount.py --batch_size 128 --target 3 --model NestedGIN_eff --h 2 --lr 4e-3 --dataset count_graphlet
python run_graphcount.py --batch_size 32 --target 4 --model NestedGIN_eff --h 4 --lr 5e-3 --dataset count_graphlet
```

QM9, you can change the target  from 0 to 11:

```
python run_qm9.py --target 0
```

ZINC:

```
python run_zinc.py --model NestedGIN_eff --layers 5 --lr 5e-4
```

OGBG-MOLHIV:

```
python run_ogb_mol.py --h 4 --num_layer 6 --save_appendix _h4_l6_spd_rd --dataset ogbg-molhiv --node_label spd --use_rd --drop_ratio 0.65 --runs 10 --efficient True --lr 1e-3 --self_loop True --edge_nest True
```

OGBG-MOLPCBA:

```
python run_ogb_mol.py --h 3 --num_layer 4 --save_appendix _h3_l4_spd_rd_edge_self_eff --dataset ogbg-molpcba --use_rd --drop_ratio 0.5 --epochs 150 --runs 10 --edge_nest True --self_loop True --efficient True --lr 2e-4
```

For evaluating the expressiveness in terms of differentiating non-isomorphic graphs:

```
python run_sr.py
python run_csl.py
python run_exp.py
```

 



The new experiments on generating cycles on ZINC, you can change the target from 0 to 3:

```
python run_zinc_cycle.py --model NestedGIN_eff --h 3 --target 0
```



Subgraphs can be generated in parallel now using package "pqdm" or "concurrent.futures", details please see dataset_zinc.py. You can simply set "num_workers" to be larger than 1 to obtain a parallel generation.
