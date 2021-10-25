"""
**************
Tree Augmented 
Naive Bayes
**************

TAN is considered to be quite useful
as a Bayesian network classifier.

"""
from rpi_d3m_primitives.pyBN.learning.structure.tree.chow_liu import chow_liu
from rpi_d3m_primitives.pyBN.classes.bayesnet import BayesNet
import numpy as np

def TAN(data, target):
    """
    Learn a Tree-Augmented Naive Bayes structure
    from data.

    The algorithm from Friedman's paper
    proceeds as follows:

    - Learn a tree structure
        - I will use chow-liu algorithm
    - ADD a class label C to the graph, and an edge
    from C to each node in the graph.
    """
    reduced_data = data[:,:target]
    tree, value_dict = chow_liu(reduced_data,edges_only=True)
    temp = dict()
    temp[target] = list(tree.keys())
    tree.update(temp)

    temp = dict()
    temp[target] = list(np.unique(data[:,target]))
    value_dict.update(temp)

    bn = BayesNet(tree, value_dict)
    return bn
