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
import csv

from .dataset import ClassificationTaskDataset
from .expansion import MetaclassClassificationTaskExpander


class CUBTasks:
    """CUBTasks generates tasks from the CUB dataset

    Parameters
    ----------
    cub_dataset : CUBDataset
    """
    TAXONOMY_FILE = 'taxonomy.txt'

    ORDER_TASK = 'order'
    FAMILY_TASK = 'family'
    GENUS_TASK = 'genus'
    SPECIES_TASK = 'species'
    POSSIBLE_TASKS = [ORDER_TASK, FAMILY_TASK, GENUS_TASK, SPECIES_TASK]
    TAXONOMY_COLUMN_MAP = {ORDER_TASK: 2, FAMILY_TASK: 3, GENUS_TASK: 4, SPECIES_TASK: 5}

    def __init__(self, cub_dataset):
        self.cub_dataset = cub_dataset

    def generate(self, task='order', task_id=None, taxonomy_file=TAXONOMY_FILE,
                 use_species_names=False, transform=None, target_transform=None):
        """Generate tasks given the specified task (order|family|genus|species)

        Parameters
        ----------
        task: str (default 'order')
            tasks to generate
        task_id : int or None (default 0)
            if None, generate all tasks otherwise only generate for task_id
        taxonomy_file : str
            taxonomy file name if provided
        use_species_names : bool (default False)
            if True, use species name in taxonomy file instead of default label names
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : task id -> Task if task_id is None
            or
        Task if task_id is set
        """
        assert isinstance(task, str)
        assert task in CUBTasks.POSSIBLE_TASKS, task

        taxonomy_path = os.path.join(self.cub_dataset.root, taxonomy_file)
        assert os.path.exists(taxonomy_path), taxonomy_path

        task_map = []
        task_col_index = CUBTasks.TAXONOMY_COLUMN_MAP[task]
        task_names_to_ids = {}
        with open(taxonomy_path, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                label = int(row[0])
                label_name = row[5].replace('_', ' ') if use_species_names else row[1]
                task_name = row[task_col_index]

                if task_name not in task_names_to_ids:
                    new_task_id = len(task_map)
                    task_names_to_ids[task_name] = new_task_id
                    task = {
                        "task_id": new_task_id,
                        "task_name": task_name,
                        "class_names": [label_name],
                        "class_ids": [0],
                        "label_map": {label: 0},
                    }
                    task_map.append(task)
                else:
                    new_task_id = task_names_to_ids[task_name]
                    new_label = len(task_map[new_task_id]['label_map'])
                    task_map[new_task_id]['class_names'].append(label_name)
                    task_map[new_task_id]['class_ids'].append(new_label)
                    task_map[new_task_id]['label_map'].update({label: new_label})

        task_expander = MetaclassClassificationTaskExpander()
        if task_id is not None:
            assert isinstance(task_id, int)
            assert 0 <= task_id < len(task_map)
            task_map = [task_map[task_id]]
            results = task_expander(self.cub_dataset, task_map,
                                    transform=transform,
                                    target_transform=target_transform)
            assert len(results) == 1, len(results)
            return list(results.values())[0]
        else:
            return task_expander(self.cub_dataset, task_map,
                                 transform=transform,
                                 target_transform=target_transform)


class CUBDataset(ClassificationTaskDataset):
    """CUB Dataset

    Parameters
    ----------
    path : str (default None)
        path to dataset (should contain images folder in same directory)
        if None, search using DATA environment variable
    split : str (train|test) or None (default 'train')
        only load split if provided, otherwise if None, load train+test
    classes_file : str
        path to class names file (relative to path argument)
    metadata : dict (default empty)
        extra arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    """
    IMAGES_FOLDER = 'images'
    IMAGES_FILE = 'images.txt'
    TRAIN_TEST_SPLIT_FILE = 'train_test_split.txt'
    IMAGE_CLASS_LABELS_FILE = 'image_class_labels.txt'
    CLASSES_FILE = 'classes.txt'

    TRAIN_SPLIT = 'train'
    TEST_SPLIT = 'test'
    POSSIBLE_SPLITS = [TRAIN_SPLIT, TEST_SPLIT]

    def __init__(self, root, split='train', classes_file=CLASSES_FILE,
                 metadata={}, transform=None, target_transform=None):

        path = os.path.join(root, 'cub/CUB_200_2011')
        images_folder = os.path.join(path, CUBDataset.IMAGES_FOLDER)
        images_file = os.path.join(path, CUBDataset.IMAGES_FILE)
        train_test_split_file = os.path.join(path, CUBDataset.TRAIN_TEST_SPLIT_FILE)
        image_class_labels_file = os.path.join(path, CUBDataset.IMAGE_CLASS_LABELS_FILE)
        classes_file = os.path.join(path, classes_file)

        assert os.path.exists(images_folder), images_folder
        assert os.path.exists(images_file), images_file
        assert os.path.exists(image_class_labels_file), image_class_labels_file

        # read in splits
        ignore_indices = set()
        if split is not None:
            assert split in CUBDataset.POSSIBLE_SPLITS, split
            assert os.path.exists(train_test_split_file), train_test_split_file

            with open(train_test_split_file, 'r') as f:
                for l in f:
                    index, is_train = l.strip().split(' ')
                    if int(is_train) and split == CUBDataset.TEST_SPLIT:
                        ignore_indices.add(int(index))
                    elif not int(is_train) and split == CUBDataset.TRAIN_SPLIT:
                        ignore_indices.add(int(index))

        # read in images
        images_list = []
        with open(images_file, 'r') as f:
            for l in f:
                index, img = l.strip().split(' ')
                if int(index) not in ignore_indices:
                    images_list.append(os.path.join(CUBDataset.IMAGES_FOLDER, img))

        # read in labels
        labels_list = []
        with open(image_class_labels_file, 'r') as f:
            for l in f:
                index, label = l.strip().split(' ')
                if int(index) not in ignore_indices:
                    labels_list.append(int(label))

        # read in label names
        label_names = {}
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                for l in f:
                    label, label_name = l.strip().split(' ', 1)
                    label_names[int(label)] = label_name

        self.split = split
        super(CUBDataset, self).__init__(images_list,
                                         labels_list,
                                         label_names=label_names,
                                         root=path,
                                         task_id=None,
                                         task_name='CUB',
                                         metadata=metadata,
                                         transform=transform,
                                         target_transform=target_transform)
