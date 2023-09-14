import numpy as np
from opt.defaults.modifier import ClusterSplitMerge

def test_default_modifier_ClusterSplit():
    """
    Unit test for evaluating the `eval` method of the `ClusterSplitMerge` class
    """

    modifier = ClusterSplitMerge(split_prob=1.0, merge_prob=0.0)

    """ Test case with two clusters of even-numbers of nodes within each"""
    cluster_assignments = [[0, 1], [2, 3]]
    modded_assignments = modifier.modify(cluster_assignments)

    assert len(modded_assignments) == 3

    """ Test case where there is one cluster with only 1 node -- modded assignments should not be split """
    cluster_assignments = [[0]]
    modded_assignments = modifier.modify(cluster_assignments)

    assert len(modded_assignments) == 1

    """ Test another scase with one cluster containing an odd-number of nodes"""
    cluster_assignments = [[0, 1, 2]]
    modded_assignments = modifier.modify(cluster_assignments)

    lengths = np.sort(np.array([len(assgn_i) for assgn_i in modded_assignments]))
    assert np.allclose(lengths, np.array([1, 2]))

def test_default_modifier_ClusterMerge():
    """
    Unit test for evaluating the `eval` method of the `ClusterSplitMerge` class
    """

    modifier = ClusterSplitMerge(split_prob=0.0, merge_prob=1.0)

    """ Test case with two clusters of even-numbers of nodes within each"""
    cluster_assignments = [[0, 1], [2, 3]]
    modded_assignments = modifier.modify(cluster_assignments)

    assert len(modded_assignments) == 1
    assert len(modded_assignments[0]) == 4

    """ Test case where there is one cluster -- modded assignments should be the same """
    cluster_assignments = [[0, 1, 2]]
    modded_assignments = modifier.modify(cluster_assignments)

    assert len(modded_assignments) == 1
    assert len(modded_assignments[0]) == 3

    """ Test case with two clusters, where one cluster only has one node"""
    cluster_assignments = [[0, 1], [3]]
    modded_assignments = modifier.modify(cluster_assignments)

    assert len(modded_assignments) == 1
    assert len(modded_assignments[0]) == 3

# def test_default_modifier_():
#     pass
