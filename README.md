# SC-GAT

Structural Coordinate Based GAT

As part of my research I stumpled across a paper [Pre-training of Graph Neural Network for Modeling Effects of Mutations on Protein-Protein Binding Affinity](https://arxiv.org/abs/2008.12473) that solves my problem. Kind of. Upon trying to replicate their work and maybe find some improvements, I have found that their code is kind of ugly, and not well documented. This in itself wouldnt impose a problem to replication, but the fact that it just doesn't work right does. [Link to their project](https://github.com/Liuxg16/GeoPPI/).

So up to this point I have explored their idea and my goal is to improve their work, and add some of my own ideas.

This particular repository is for the special type of Graph Attention Network that works with relative coordinates instead of the usual form of GATs. I will put the corresoponding equations in a Jupyter Notebook so it makes more sense. Although they have an [implementation](https://github.com/Liuxg16/GeoPPI/blob/master/cgat.py), it doesn't work and things like edge attributes aren't even used, so I am really dissatisfied overall. 


I think they have tried to implement it on their own instead of just reusing the torch_geometric GAT implementation(s). Using that makes it so easy that, I feel kind of dumb basically posting code of others with 3 lines of change, but let's concentrate on the ideas behind it. The GATv2Conv source code can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gatv2_conv.html#GATv2Conv), I took the liberty to just update it with my own definition of attention weights. I will probably try some other ideas also.

Immediately we see how important this is for molecules.

Proper explanation of how this NN layer looks can be found in the test_gat.ipynb notebook.


## Using SCGAT 

You just need torch-geometric and pytorch.
