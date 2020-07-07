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

import os

from torchvision.datasets import MNIST
import torch.nn.functional as F

from .dataset import ClassificationTaskDataset
from .expansion import ClassificationTaskExpander

class SplitMNISTTask:
    """SplitMNISTTask generates Split MNIST tasks given two classes

    Parameters
    ----------
    mnist_dataset : MNISTDataset
    """
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def generate(self, classes=(0, 1), transform=None, target_transform=None):
        """Generate tasks given the classes

        Parameters
        ----------
        classes : tuple of ints (0-9)
            two classes to generate split MNIST
            (possible to accept more than two classes)
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        Task
        """
        assert isinstance(classes, tuple)
        assert all(isinstance(c, int) and c >= 0 and c <= 9 for c in classes)
        assert len(set(classes)) == len(classes)

        task_expander = ClassificationTaskExpander()
        return task_expander(self.mnist_dataset,
                             {c : new_c for new_c, c in enumerate(classes)},
                             label_names={classes.index(old_c) : name for old_c, name in self.mnist_dataset.label_names_map.items() if old_c in classes},
                             task_id=classes,
                             task_name='Split MNIST {}'.format(classes),
                             transform=transform,
                             target_transform=target_transform)

class MNISTDataset(ClassificationTaskDataset):
    """MNIST Dataset

    Parameters
    ----------
    path : str (default None)
        path to dataset (should contain images folder in same directory)
        if None, search using DATA environment variable
    train : bool (default True)
        if True, load train split otherwise load test split
    download: bool (default False)
        if True, downloads the dataset from the internet and
        puts it in path directory; otherwise if dataset is already downloaded,
        it is not downloaded again
    metadata : dict (default empty)
        extra arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    expand: pad images to 32x32 and expand them to have 3  images to be
    """
    def __init__(self, path=None, train=True, download=False,
                 metadata={}, transform=None, target_transform=None, expand=False):
        if path is not None:
            assert isinstance(path, str)
        path = os.path.join(os.environ['DATA'], 'mnist') if path is None else path
        assert isinstance(train, bool)

        self.mnist = MNIST(path, train=train, transform=transform,
                           target_transform=target_transform, download=download)
        data = self.mnist.train_data if train else self.mnist.test_data
        labels = self.mnist.train_labels if train else self.mnist.test_labels

        if expand:
            data = F.pad(data, [2,2,2,2])
            data = data.view(-1, 32, 32, 1).expand([data.size(0), 32, 32, 3])

        super(MNISTDataset, self).__init__([x for x in data],
                                           [int(x) for x in labels],
                                           label_names={l: str(l) for l in range(10)},
                                           root=path,
                                           task_id=None,
                                           task_name='MNIST',
                                           metadata=metadata,
                                           transform=transform,
                                           target_transform=target_transform)
