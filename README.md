# SC-GAT

Structural Coordinate Based GAT

As part of my research I found a paper [Pre-training of Graph Neural Network for Modeling Effects of Mutations on Protein-Protein Binding Affinity](https://arxiv.org/abs/2008.12473) that aimes to resolve my problem. Upon trying to replicate their work and maybe find some improvements I found their code contains some errors and is not well documented. [Link to their project](https://github.com/Liuxg16/GeoPPI/).

So up to this point I have explored their idea and my goal is to improve their work, and add some of my own ideas.

This particular repository is for the special type of Graph Attention Network that works with relative coordinates instead of the usual form of GATs. I will put the corresoponding equations in a Jupyter Notebook so it makes more sense. Although they have an [implementation](https://github.com/Liuxg16/GeoPPI/blob/master/cgat.py), it doesn't work and things like edge attributes aren't even used.

The GATv2Conv source code can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv), I took the liberty to just update it with my own definition of attention weights. I will probably try some other ideas also.

Proper explanation of how this NN layer looks can be found in the test_gat.ipynb notebook.

## Using SCGAT 

You just need torch-geometric and pytorch.
