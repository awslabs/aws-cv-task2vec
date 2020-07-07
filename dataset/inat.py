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
from collections import defaultdict
import json

from .dataset import ClassificationTaskDataset


class iNat2018Dataset(ClassificationTaskDataset):
    """iNaturalist 2018 Dataset

    Parameters
    ----------
    path : str (default None)
        path to dataset
        if None, search using DATA environment variable
    split : str (train|test)
        load provided split
    task_id : int (default 0)
        id of task
    level : str (default 'order')
        what level of taxonomy to create tasks
    metadata : dict (default empty)
        extra arbitrary metadata
    transform : callable (default None)
        Optional transform to be applied on a sample.
    target_transform : callable (default None)
        Optional transform to be applied on a label.
    """
    CATEGORIES_FILE = 'categories.json'
    TASKS_FILE = 'tasks.json'
    CLASS_TASKS_FILE = 'classes_tasks.json'
    TRAIN_2018_FILE = 'train2018.json'

    ORDER = 'order'
    CLASS = 'class'
    POSSIBLE_LEVELS = [ORDER, CLASS]

    TRAIN_SPLIT = 'train'
    VAL_SPLIT = 'val'
    POSSIBLE_SPLITS = [TRAIN_SPLIT, VAL_SPLIT]

    def __init__(self, root, split=TRAIN_SPLIT, task_id=0, level=ORDER,
                 metadata={}, transform=None, target_transform=None):
        assert isinstance(root, str)
        path = os.path.join(root, 'inat2018')
        assert os.path.exists(path), path
        assert split in iNat2018Dataset.POSSIBLE_SPLITS, split
        assert isinstance(task_id, int)
        assert level in iNat2018Dataset.POSSIBLE_LEVELS, level

        self.split = split

        # load categories
        with open(os.path.join(path, iNat2018Dataset.CATEGORIES_FILE)) as f:
            self.categories = json.load(f)

        # load or create set of tasks (corresponding to classification inside orders)
        tasks_path = os.path.join(path, iNat2018Dataset.TASKS_FILE) if level == 'order' else os.path.join(path,
                                                                                                          iNat2018Dataset.CLASS_TASKS_FILE)
        if not os.path.exists(tasks_path):
            self._create_tasks(tasks_path, path, level=level)
        with open(tasks_path) as f:
            self.tasks = json.load(f)

        annotation_file = os.path.join(path, "{}2018.json".format(split))
        assert os.path.exists(annotation_file), annotation_file
        with open(annotation_file) as f:
            j = json.load(f)
        annotations = j['annotations']
        images_list = [img['file_name'] for img in j['images']]

        # get labels
        task = self.tasks[task_id]
        species_ids = task['species_ids']
        species_ids = {k: i for i, k in enumerate(species_ids)}
        labels_list = [species_ids.get(a['category_id'], -1) for a in annotations]

        # throw away images we are not training on (identified by label==-1)
        images_list = [img for i, img in enumerate(images_list) if labels_list[i] != -1]
        labels_list = [l for l in labels_list if l != -1]

        # get label names
        label_names = {task['label_map'][str(l)]: n for l, n in zip(task['species_ids'], task['species_names'])}

        task_name = self.tasks[task_id]['name']
        super(iNat2018Dataset, self).__init__(images_list,
                                              labels_list,
                                              label_names=label_names,
                                              root=path,
                                              task_id=task_id,
                                              task_name=task_name,
                                              metadata=metadata,
                                              transform=transform,
                                              target_transform=target_transform)

    def _create_tasks(self, tasks_path, root, level=ORDER):
        """Create tasks file.

        Post-conditions:
            Creates a file

        Parameters
        ----------
        tasks_path : str
            path to tasks file to write
        root : str
            root folder to train file
        level : str (default 'order')
            what level of taxonomy to create tasks
        """
        assert level in iNat2018Dataset.POSSIBLE_LEVELS, level

        with open(os.path.join(root, iNat2018Dataset.TRAIN_2018_FILE)) as f:
            annotations = json.load(f)
        annotations = annotations['annotations']

        level_samples = defaultdict(list)
        for r in annotations:
            level_samples[self.categories[r['category_id']][level]].append(r['image_id'])

        tasks = []
        for i, (level_name, _) in enumerate(sorted(level_samples.items(), key=lambda x: len(x[1]), reverse=True)):
            species_in_level = sorted([(c['name'], c['id']) for c in self.categories if c[level] == level_name],
                                      key=lambda x: x[0])
            species_ids = {k: i for i, (_, k) in enumerate(species_in_level)}
            tasks.append({
                'id': i,
                'name': level_name,
                'species_names': [n for (n, i) in species_in_level],
                'species_ids': [i for (n, i) in species_in_level],
                'label_map': species_ids
            })
        with open(tasks_path, 'w') as f:
            json.dump(tasks, f)
