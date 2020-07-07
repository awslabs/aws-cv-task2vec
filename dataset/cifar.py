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

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from .dataset import ClassificationTaskDataset
from .expansion import ClassificationTaskExpander


class SplitCIFARTask:
    """SplitCIFARTask generates Split CIFAR task

    Parameters
    ----------
    cifar10_dataset : CIFAR10Dataset
    cifar100_dataset : CIFAR100Dataset
    """

    def __init__(self, cifar10_dataset, cifar100_dataset):
        self.cifar10_dataset = cifar10_dataset
        self.cifar100_dataset = cifar100_dataset

    def generate(self, task_id=0, transform=None, target_transform=None):
        """Generate tasks given the classes

        Parameters
        ----------
        task_id : int 0-10 (default 0)
            0 = CIFAR10, 1 = first 10 of CIFAR100, 2 = second 10 of CIFAR100, ...
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        Task
        """
        assert isinstance(task_id, int)
        assert 0 <= task_id <= 10, task_id

        task_expander = ClassificationTaskExpander()
        if task_id == 0:
            classes = tuple(range(10))
            return task_expander(self.cifar10_dataset,
                                 {c: new_c for new_c, c in enumerate(classes)},
                                 label_names={c: name for c, name in self.cifar10_dataset.label_names_map.items()},
                                 task_id=task_id,
                                 task_name='Split CIFAR: CIFAR-10 {}'.format(classes),
                                 transform=transform,
                                 target_transform=target_transform)
        else:
            classes = tuple([int(c) for c in np.arange(10) + 10 * (task_id - 1)])
            return task_expander(self.cifar100_dataset,
                                 {c: new_c for new_c, c in enumerate(classes)},
                                 label_names={classes.index(old_c): name for old_c, name in
                                              self.cifar100_dataset.label_names_map.items() if old_c in classes},
                                 task_id=task_id,
                                 task_name='Split CIFAR: CIFAR-100 {}'.format(classes),
                                 transform=transform,
                                 target_transform=target_transform)


class CIFAR10Dataset(ClassificationTaskDataset):
    """CIFAR10 Dataset

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
    """

    def __init__(self, path, train=True, download=False,
                 metadata={}, transform=None, target_transform=None):
        num_classes, task_name = self._get_settings()
        assert isinstance(path, str)
        assert isinstance(train, bool)

        self.cifar = self._get_cifar(path, train, transform, target_transform, download)

        super(CIFAR10Dataset, self).__init__(list(self.cifar.data),
                                             [int(x) for x in self.cifar.targets],
                                             label_names={l: str(l) for l in range(num_classes)},
                                             root=path,
                                             task_id=None,
                                             task_name=task_name,
                                             metadata=metadata,
                                             transform=transform,
                                             target_transform=target_transform)

    def _get_settings(self):
        return 10, 'CIFAR10'

    def _get_cifar(self, path, train, transform, target_transform, download=True):
        return CIFAR10(path, train=train, transform=transform,
                       target_transform=target_transform, download=download)


class CIFAR100Dataset(CIFAR10Dataset):
    """CIFAR100 Dataset

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
    """

    def __init__(self, path=None, train=True, download=False,
                 metadata={}, transform=None, target_transform=None):
        super(CIFAR100Dataset, self).__init__(path=path,
                                              train=train,
                                              metadata=metadata,
                                              transform=transform,
                                              target_transform=target_transform)

    def _get_settings(self):
        return 100, 'CIFAR100'

    def _get_cifar(self, path, train, transform, target_transform, download=True):
        return CIFAR100(path, train=train, transform=transform,
                        target_transform=target_transform, download=download)
