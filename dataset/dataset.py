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
import sys
import collections
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from sklearn.preprocessing import MultiLabelBinarizer


def is_multi_label(labels_list):
    """Whether labels list provided is a multi-label dataset.

    Parameters
    ----------
    labels_list : list
        list of labels (integers) or multiple labels (lists of integers) or mixed

    Returns
    -------
    True if multi-label
    """
    return any([isinstance(l, list) for l in labels_list])


class MultilabelTransform:
    """MultilabelTransform transforms a list of labels into a multilabel format

    E.g. if possible_labels=[1,2,3,4,5], then
            sample=[3,5] => [0,0,1,0,1]

    Parameters
    ----------
    possible_labels : list
        An ordering for the class labels
    """

    def __init__(self, possible_labels):
        assert isinstance(possible_labels, list)
        assert len(possible_labels) > 0
        assert len(possible_labels) == len(set(possible_labels))
        self._transformer = MultiLabelBinarizer(classes=possible_labels)

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample : target label

        Returns
        -------
        target label in multilabel format
        """
        if isinstance(sample, int):
            return self._transformer.fit_transform([[sample]])[0]
        else:  # list
            return self._transformer.fit_transform([sample])[0]


class TaskDataset(data.Dataset):
    """TaskDataset allows task expansion operations and is initialized with a list of images and labels.

    Parameters
    ----------
    images_list : list
        list of image paths (str)
    labels_list : list
        list of labels
    root : str
        root path to append to all images
    task_id : int or tuple, default None
        task id (simply used as metadata)
    task_name : str, default None
        task name (simply used as metadata)
    metadata : dict (default empty)
        extra arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    """

    def __init__(self, images_list, labels_list, root='',
                 task_id=None, task_name=None, metadata={},
                 transform=None, target_transform=None):
        assert isinstance(images_list, list)
        assert isinstance(labels_list, list)
        assert len(images_list) == len(labels_list)
        assert isinstance(root, str)
        if task_id is not None:
            assert isinstance(task_id, int) or isinstance(task_id, tuple)
        if task_name is not None:
            assert isinstance(task_name, str)
        assert isinstance(metadata, dict)

        self._images = [img for img in images_list]
        self._labels = list(labels_list)
        self._root = root

        self._task_id = task_id
        self._task_name = task_name
        self._metadata = deepcopy(metadata)

        self._transform = transform
        self._target_transform = target_transform

        self._loader = default_loader

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        img : image
            sample
        target : int
            class_index of target class
        """
        if isinstance(self.images[index], str):
            img_path = self.images[index]
            try:
                img = self._loader(os.path.join(self._root, img_path))
            except OSError as ex:
                # If the file cannot be read, print error and return grey image instead
                print(ex, file=sys.stderr)
                img = Image.fromarray(np.ones([224, 224, 3], dtype=np.uint8) * 128)

        elif isinstance(self.images[index], torch.Tensor):
            img = Image.fromarray(self.images[index].numpy())
        elif isinstance(self.images[index], np.ndarray):
            img = Image.fromarray(self.images[index])
        else:
            raise NotImplementedError()
        target = self.labels[index]
        if self._transform is not None:
            img = self._transform(img)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return img, target

    @property
    def images(self):
        """List of images.
        """
        return self._images

    @property
    def labels(self):
        """List of labels, corresponding to the list of images.
        """
        return self._labels

    @property
    def possible_labels(self):
        """List of possible labels.

        Returns
        -------
        list of labels (int)
        """
        labels = set()
        for label in self.labels:
            if isinstance(label, collections.Iterable):
                labels.update([l for l in label])
            else:
                labels.add(label)
        return sorted(list(labels))

    @property
    def num_classes(self):
        return len(self.possible_labels)

    @property
    def task_id(self):
        return self._task_id

    @property
    def task_name(self):
        return self._task_name

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = metadata

    @property
    def root(self):
        return self._root

    @property
    def transform(self):
        return self._transform

    @property
    def target_transform(self):
        return self._target_transform


class ClassificationTaskDataset(TaskDataset):
    """ClassificationTaskDataset allows task expansion operations for classification tasks.

    Parameters
    ----------
    images_list : list
        list of image paths (str)
    labels_list : list
        list of labels (integers) or multiple labels (lists of integers)
        (old task)
    label_names : dict (default None)
        map of label (int) -> name (str)
        if task_mapper is not None, should be using new labels
    binarize_labels : bool (default False)
        if True, binarize labels in dataset iterator (should be used for multilabel)
    force_multi_label : bool (default False)
        if True, force multi-label dataset
    root : str
        root path to append to all images
    task_id : int, default None
        task id (simply used as metadata)
    task_name : str, default None
        task name (simply used as metadata)
    metadata : dict (default empty)
        arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    """

    def __init__(self, images_list, labels_list, label_names=None, root='',
                 binarize_labels=False, force_multi_label=False,
                 task_id=None, task_name=None, metadata={},
                 transform=None, target_transform=None):
        assert isinstance(labels_list, list)
        assert all(isinstance(x, int) or isinstance(x, list) for x in labels_list)
        assert all(all(isinstance(i, int) for i in x) if isinstance(x, list) else True for x in labels_list)

        # force multi-label format when at least one label instance is multi-label
        if force_multi_label or is_multi_label(labels_list):
            labels_list = [l if isinstance(l, list) else [l] for l in labels_list]

        super(ClassificationTaskDataset, self).__init__(images_list,
                                                        labels_list, root=root,
                                                        task_id=task_id, task_name=task_name, metadata=metadata,
                                                        transform=transform, target_transform=target_transform)

        # sanity checks for label names
        if label_names is not None:
            self._verify_label_names(label_names)
        self._label_names_map = label_names

        if binarize_labels or self.is_multi_label:
            target_transforms = [
                MultilabelTransform(self.possible_labels),
            ]
            if target_transform is not None:
                target_transforms.append(target_transform)
            self._target_transform = transforms.Compose(target_transforms)

    def _verify_label_names(self, label_names):
        """Verify label names given labels.

        Raises an exception if label names are not proper with respect to the given labels.

        Parameters
        ----------
        label_names : dict
            map of label -> name
        """
        assert isinstance(label_names, dict)

        found_labels = set()
        for label in self._labels:
            if isinstance(label, collections.Iterable):
                found_labels.update([l for l in label])
            else:
                found_labels.add(label)
        assert found_labels.issubset(
            label_names.keys()), 'dataset contains labels not specified in label_names: {} vs. {}'.format(found_labels,
                                                                                                          label_names.keys())
        if len(found_labels) < len(label_names.keys()):
            print("Warning: label_names contains more labels than discovered labels in the dataset")

    def get_labels(self, flatten=False):
        """List of labels, corresponding to the list of images.

        Parameters
        ----------
        flatten : bool (default False)
            flattens list into all labels (destroys correspondence with images)
            comes in handy for calculating label frequency
        """
        if flatten:
            flattened_labels_list = []
            for l in self._labels:
                if isinstance(l, list):
                    flattened_labels_list.extend(l)
                else:
                    flattened_labels_list.append(l)
            return flattened_labels_list
        else:
            return self._labels

    @property
    def is_multi_label(self):
        """Whether dataset is a multi-label dataset.

        Returns
        -------
        True if multi-label dataset
        """
        return is_multi_label(self._labels)

    @property
    def label_names(self):
        """List of label names for each sample.

        Returns
        -------
        list of label names per sample (str)
        """
        if not self._label_names_map:
            return None
        else:
            return [[self._label_names_map[l] for l in x] if type(x) == list else self._label_names_map[x] for x in
                    self.labels]

    @property
    def possible_label_names(self):
        """List of possible label names.

        Returns
        -------
        list of label names (str)
        """
        if not self._label_names_map:
            return None
        else:
            return [self._label_names_map[l] for l in self.possible_labels]

    @property
    def label_names_map(self):
        """Map of label to label name.
        """
        return self._label_names_map
