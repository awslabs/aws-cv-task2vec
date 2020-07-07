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
import json
import csv

from .dataset import ClassificationTaskDataset
from .expansion import BinaryClassificationTaskExpander, MetaclassClassificationTaskExpander


class iMat2018FashionTasks:
    """iMat2018FashionTasks generates binary attributes tasks from the iMaterialist 2018 fashion dataset

    Parameters
    ----------
    imat_dataset : iMat2018FashionDataset
    """
    LABEL_MAP_FILE = 'labels.csv'
    CUSTOM_CATEGORY_TYPES = ['pants', 'shoes', 'other']
    CATEGORY_TASK_ID = 1
    ATTRIBUTE_TASK_IDS = [2, 3, 4, 5, 6, 8, 9]

    def __init__(self, imat_dataset):
        self.dataset = imat_dataset
        self._setup_label_map()

    def _setup_label_map(self):
        """Helper to load annotation data.
        """
        label_map_path = os.path.join(self.dataset.root, iMat2018FashionTasks.LABEL_MAP_FILE)
        self.label_map = None
        if not os.path.exists(label_map_path):
            return

        self.label_map = {}
        with open(label_map_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx < 1:
                    continue
                label_id, task_id, label_name, task_name, is_pants, is_shoe = row
                self.label_map[label_id] = row

    def generate(self, task_id=0, transform=None, target_transform=None):
        return self.generate_from_attributes(task_id=task_id,
                                             transform=transform,
                                             target_transform=target_transform)

    def generate_from_attributes(self, task_id=0, transform=None, target_transform=None):
        """Generate binary attributes tasks.

        Parameters
        ----------
        task_id : int or None (default 0)
            if None, generate all tasks otherwise only generate for task_id
            task_id is iMat label-1 (i.e. zero-indexed)
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : label (int) -> ClassificationTaskDataset
        """
        task_expander = BinaryClassificationTaskExpander()
        if task_id is not None:
            assert isinstance(task_id, int)
            label = task_id + 1
            assert label in self.dataset.possible_labels, label
            results = task_expander(self.dataset, labels=[label],
                                    transform=transform,
                                    target_transform=target_transform)
            assert len(results) == 1, len(results)
            return list(results.values())[0]
        else:
            return task_expander(self.dataset,
                                 transform=transform,
                                 target_transform=target_transform)

    def generate_from_custom_category_types(self, task_id=0,
                                            transform=None, target_transform=None):
        """Generate from 3 custom category tasks: pants, shoes, other.

        Parameters
        ----------
        task_id : int or None (default 0)
            if None, generate all tasks otherwise only generate for task_id
            task_id is 0-2: pants, shoes, other
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : label (int) -> ClassificationTaskDataset
        """
        # initialize task map
        task_map = []
        for idx, task_name in enumerate(iMat2018FashionTasks.CUSTOM_CATEGORY_TYPES):
            task_map.append({
                "task_id": idx,
                "task_name": task_name,
                "class_names": [],
                "class_ids": [],
                "label_map": {},
            })

        # build task map
        for idx, (label_id, data) in enumerate(self.label_map.items()):
            label_id, data_task_id, label_name, task_name, is_pants, is_shoe = data
            label_id = int(label_id)
            data_task_id = int(data_task_id)
            is_pants = is_pants == 'yes'
            is_shoe = is_shoe == 'yes'
            if data_task_id == iMat2018FashionTasks.CATEGORY_TASK_ID:
                if is_pants:
                    task_map[0]['class_names'].append(label_name)
                    new_label_index = len(task_map[0]['class_ids'])
                    task_map[0]['class_ids'].append(new_label_index)
                    task_map[0]['label_map'].update({label_id: new_label_index})
                elif is_shoe:
                    task_map[1]['class_names'].append(label_name)
                    new_label_index = len(task_map[1]['class_ids'])
                    task_map[1]['class_ids'].append(new_label_index)
                    task_map[1]['label_map'].update({label_id: new_label_index})
                else:
                    task_map[2]['class_names'].append(label_name)
                    new_label_index = len(task_map[2]['class_ids'])
                    task_map[2]['class_ids'].append(new_label_index)
                    task_map[2]['label_map'].update({label_id: new_label_index})

        # create tasks
        # note: "other" category has at least one bad sample with multiple labels; remove it
        task_expander = MetaclassClassificationTaskExpander()
        if task_id is not None:
            assert isinstance(task_id, int)
            assert task_id in range(len(task_map)), task_id
            results = task_expander(self.dataset, [task_map[task_id]],
                                    force_remove_multi_label=(task_id == 2),
                                    transform=transform,
                                    target_transform=target_transform)
            assert len(results) == 1, len(results)
            return list(results.values())[0]
        else:
            results = task_expander(self.dataset, task_map[0:2],
                                    transform=transform,
                                    target_transform=target_transform)
            other_results = task_expander(self.dataset, [task_map[2]],
                                          force_remove_multi_label=True,
                                          transform=transform,
                                          target_transform=target_transform)
            results.update(other_results)
            return results

    def generate_from_attribute_types(self, task_id=0,
                                      transform=None, target_transform=None):
        """Generate from attribute types.

        Parameters
        ----------
        task_id : int or None (default 0)
            if None, generate all tasks otherwise only generate for task_id
            task_id is 0-6: color, gender, material, neckline, pattern, sleeve, style
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : label (int) -> ClassificationTaskDataset
        """
        # initialize task map
        task_map = []
        task_ids_map = {data_task_id: idx for idx, data_task_id in enumerate(iMat2018FashionTasks.ATTRIBUTE_TASK_IDS)}
        for idx, data_task_id in enumerate(iMat2018FashionTasks.ATTRIBUTE_TASK_IDS):
            task_map.append({
                "task_id": data_task_id,
                "task_name": None,
                "class_names": [],
                "class_ids": [],
                "label_map": {},
                "multi_label": 1,
            })

        # build task map
        for idx, (label_id, data) in enumerate(self.label_map.items()):
            label_id, data_task_id, label_name, task_name, is_pants, is_shoe = data
            label_id = int(label_id)
            data_task_id = int(data_task_id)
            is_pants = is_pants == 'yes'
            is_shoe = is_shoe == 'yes'
            if data_task_id in iMat2018FashionTasks.ATTRIBUTE_TASK_IDS:
                task_idx = task_ids_map[data_task_id]
                task_map[task_idx]['task_id'] = data_task_id
                task_map[task_idx]['task_name'] = task_name
                task_map[task_idx]['class_names'].append(label_name)
                new_label_index = len(task_map[task_idx]['class_ids'])
                task_map[task_idx]['class_ids'].append(new_label_index)
                task_map[task_idx]['label_map'].update({label_id: new_label_index})

        # create tasks
        task_expander = MetaclassClassificationTaskExpander()
        if task_id is not None:
            assert isinstance(task_id, int)
            assert task_id in range(len(task_map)), task_id
            results = task_expander(self.dataset, [task_map[task_id]],
                                    transform=transform,
                                    target_transform=target_transform)
            assert len(results) == 1, len(results)
            return list(results.values())[0]
        else:
            return task_expander(self.dataset, task_map,
                                 transform=transform,
                                 target_transform=target_transform)


class iMat2018FashionDataset(ClassificationTaskDataset):
    """iMaterialist 2018 Fashion Dataset

    Parameters
    ----------
    path : str (default None)
        path to dataset
        if None, search using DATA environment variable
    split : str (train|validation)
        load provided split
    binarize_labels : bool (default False)
        if True, binarize labels in dataset iterator
    metadata : dict (default empty)
        extra arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    """
    TRAIN_SPLIT = 'train'
    VAL_SPLIT = 'validation'
    POSSIBLE_SPLITS = [TRAIN_SPLIT, VAL_SPLIT]

    LABEL_MAP_FILE = 'labels.csv'

    def __init__(self, path=None, split=TRAIN_SPLIT, binarize_labels=False,
                 metadata={}, transform=None, target_transform=None):
        if path is not None:
            assert isinstance(path, str)
        path = os.path.join(os.environ['DATA'], 'imat2018', 'fashion') if path is None else path
        assert os.path.exists(path), path
        assert split in iMat2018FashionDataset.POSSIBLE_SPLITS, split

        self.split = split

        images_path = os.path.join(path, split)
        annotations_path = os.path.join(path, '{}.json'.format(split))
        assert os.path.isdir(images_path), images_path
        assert os.path.exists(annotations_path), annotations_path

        with open(annotations_path, 'r') as f:
            j = json.load(f)
        images_list = [os.path.join(split, '{}.jpg').format(img['imageId']) for img in j['images']]
        images_to_labels = {os.path.join(split, '{}.jpg').format(labels['imageId']): [int(l) for l in labels['labelId']]
                            for labels in j['annotations']}
        labels_list = [images_to_labels[img] for img in images_list]

        label_names = self._setup_label_map(path)

        super(iMat2018FashionDataset, self).__init__(images_list,
                                                     labels_list,
                                                     label_names=label_names,
                                                     root=path,
                                                     binarize_labels=binarize_labels,
                                                     task_id=None,
                                                     task_name='iMat2018',
                                                     metadata=metadata,
                                                     transform=transform,
                                                     target_transform=target_transform)

    def _setup_label_map(self, root):
        label_map_path = os.path.join(root, iMat2018FashionDataset.LABEL_MAP_FILE)
        if not os.path.exists(label_map_path):
            return None

        label_names = {}
        with open(label_map_path, 'r') as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx < 1:
                    continue
                label_id, task_id, label_name, task_name, is_pants, is_shoe = row
                label_names[int(label_id)] = label_name

        return label_names
