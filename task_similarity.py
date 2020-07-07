#!/usr/bin/env python3

# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import itertools
import scipy.spatial.distance as distance
import numpy as np
import copy
import pickle

_DISTANCES = {}


# TODO: Remove methods that do not perform well

def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn


def is_excluded(k):
    exclude = ['fc', 'linear']
    return any([e in k for e in exclude])


def load_embedding(filename):
    with open(filename, 'rb') as f:
        e = pickle.load(f)
    return e


def get_trivial_embedding_from(e):
    trivial_embedding = copy.deepcopy(e)
    for l in trivial_embedding['layers']:
        a = np.array(l['filter_logvar'])
        a[:] = l['filter_lambda2']
        l['filter_logvar'] = list(a)
    return trivial_embedding


def binary_entropy(p):
    from scipy.special import xlogy
    return - (xlogy(p, p) + xlogy(1. - p, 1. - p))


def get_layerwise_variance(e, normalized=False):
    var = [np.exp(l['filter_logvar']) for l in e['layers']]
    if normalized:
        var = [v / np.linalg.norm(v) for v in var]
    return var


def get_variance(e, normalized=False):
    var = 1. / np.array(e.hessian)
    if normalized:
        lambda2 = 1. / np.array(e.scale)
        var = var / lambda2
    return var


def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]


def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess


def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]


def get_scaled_hessian(e0, e1):
    h0, h1 = get_hessians(e0, e1, normalized=False)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)


def get_full_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0, kl1


def layerwise_kl(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0), get_layerwise_variance(e1)
    kl0 = []
    for var0, var1 in zip(layers0, layers1):
        kl0.append(np.sum(.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))))
    return kl0


def layerwise_cosine(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0, normalized=True), get_layerwise_variance(e1, normalized=True)
    res = []
    for var0, var1 in zip(layers0, layers1):
        res.append(distance.cosine(var0, var1))
    return res


@_register_distance
def kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return np.maximum(kl0, kl1).sum()


@_register_distance
def asymmetric_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0.sum()


@_register_distance
def jsd(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    var = .5 * (var0 + var1)
    kl0 = .5 * (var0 / var - 1 + np.log(var) - np.log(var0))
    kl1 = .5 * (var1 / var - 1 + np.log(var) - np.log(var1))
    return (.5 * (kl0 + kl1)).mean()


@_register_distance
def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return distance.cosine(h1, h2)


@_register_distance
def normalized_cosine(e0, e1):
    h1, h2 = get_variances(e0, e1, normalized=True)
    return distance.cosine(h1, h2)


@_register_distance
def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    return distance.correlation(v1, v2)


@_register_distance
def entropy(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return np.log(2) - binary_entropy(h1).mean()


def get_normalized_embeddings(embeddings, normalization=None):
    F = [1. / get_variance(e, normalized=False) if e is not None else None for e in embeddings]
    zero_embedding = np.zeros_like([x for x in F if x is not None][0])
    F = np.array([x if x is not None else zero_embedding for x in F])
    # FIXME: compute variance using only valid embeddings
    if normalization is None:
        normalization = np.sqrt((F ** 2).mean(axis=0, keepdims=True))
    F /= normalization
    return F, normalization


def pdist(embeddings, distance='cosine'):
    distance_fn = _DISTANCES[distance]
    n = len(embeddings)
    distance_matrix = np.zeros([n, n])
    if distance != 'asymmetric_kl':
        for (i, e1), (j, e2) in itertools.combinations(enumerate(embeddings), 2):
            distance_matrix[i, j] = distance_fn(e1, e2)
            distance_matrix[j, i] = distance_matrix[i, j]
    else:
        for (i, e1) in enumerate(embeddings):
            for (j, e2) in enumerate(embeddings):
                distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def cdist(from_embeddings, to_embeddings, distance='cosine'):
    distance_fn = _DISTANCES[distance]
    distance_matrix = np.zeros([len(from_embeddings), len(to_embeddings)])
    for (i, e1) in enumerate(from_embeddings):
        for (j, e2) in enumerate(to_embeddings):
            if e1 is None or e2 is None:
                continue
            distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def plot_distance_matrix(embeddings, labels=None, distance='cosine'):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt
    distance_matrix = pdist(embeddings, distance=distance)
    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    sns.clustermap(distance_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap='viridis_r')
    plt.show()




