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
import collections
from copy import deepcopy

from .dataset import TaskDataset, ClassificationTaskDataset


class TaskExpander:
    """TaskExpander is an abstract class for task expansion functions.

    In general, task expansion functions take old task(s) as input and produces new task(s).

    Implement call functions to accept old task(s) and produce new task(s).
    """
    def __call__(self, task):
        raise NotImplementedError()


class ClassificationTaskExpander(TaskExpander):
    """ClassificationTaskExpander remaps old classes to new classes.
    """
    def __call__(self, task, task_map, label_names=None, flatten_binary=False,
                 force_multi_label=False, force_remove_multi_label=False,
                 task_id=None, task_name=None, metadata=None,
                 transform=None, target_transform=None):
        """Call function.

        Parameters
        ----------
        task : ClassificationTaskDataset
            old task
        task_map : dict
            map of old label (int) -> new labels (list)
        label_names : dict (default None)
            map of new label (int) -> name (str)
        flatten_binary : bool (default False)
            if True and new label space is binary, set new label to 1 if attribute present
        force_multi_label : bool (default False)
            if True, force multi-label dataset
        force_remove_multi_label : bool (default False)
            if True, force removing samples with multiple labels
        task_id : int, default None
            task id (simply used as metadata)
        task_name : str, default None
            task name (simply used as metadata)
        metadata : dict (default empty)
            extra arbitrary metadata
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        ClassificationTaskDataset
        """
        assert isinstance(task_map, dict)
        assert all(isinstance(x, int) for x in task_map.keys())
        assert all(isinstance(x, int) or isinstance(x, list) for x in task_map.values())
        is_single_label_task = all(isinstance(x, int) for x in task_map.values())
        if flatten_binary:
            assert self._is_binary_new_task(task_map)

        if task_map is {}:
            return task

        # create new labels and filter images
        new_images = []
        new_labels = []
        for idx, (img, label) in enumerate(zip(task.images, task.labels)):
            if isinstance(label, list):
                # map old labels to new labels
                new_label = set()
                for l in label:
                    if l in task_map:
                        if type(task_map[l]) == int:
                            new_label.add(task_map[l])
                        else:
                            new_label.update(task_map[l])
                new_label = list(new_label)
                if len(new_label) > 0:
                    # enforce consistent data type
                    new_label = new_label[0] if len(new_label) == 1 and is_single_label_task else new_label
                    new_label = 1 if flatten_binary and new_label == [0,1] else new_label
                    new_images.append(img)
                    new_labels.append(new_label)
                # otherwise exclude sample

            elif isinstance(label, int):
                # map old label to new label
                if label in task_map:
                    new_label = task_map[label]
                    new_images.append(img)
                    new_labels.append(new_label)
                # otherwise if not specified in task mapper, exclude sample

            else:
                raise ValueError("label is not a list or int: {}".format(label))

        if force_remove_multi_label:
            multi_label_samples_mask = [isinstance(l, list) and len(l) != 1 for l in new_labels]
            new_images = [img for idx, img in enumerate(new_images) if not multi_label_samples_mask[idx]]
            new_labels = [l if isinstance(l, int) else l[0] for idx, l in enumerate(new_labels) if not multi_label_samples_mask[idx]]

        assert len(new_images) == len(new_labels)
        assert len(new_images) > 0

        return ClassificationTaskDataset(new_images,
                                         new_labels,
                                         label_names=label_names,
                                         force_multi_label=force_multi_label,
                                         root=task.root,
                                         task_id=task.task_id if task_id is None else task_id,
                                         task_name=task.task_name if task_name is None else task_name,
                                         metadata=task.metadata if metadata is None else metadata,
                                         transform=transform,
                                         target_transform=target_transform)

    def _is_binary_new_task(self, task_map):
        """Helper to determine if new task is a binary classification problem.

        Parameters
        ----------
        task_map : dict
            map of old label (int) -> new labels (list)

        Returns
        -------
        True if new task is binary classification problem
        """
        possible_labels = set()
        for label in task_map.values():
            if isinstance(label, int):
                possible_labels.add(label)
            elif isinstance(label, list):
                possible_labels.update(label)
            else:
                raise ValueError('Expected int or list but got {}'.format(type(label)))
        return possible_labels == set([0,1])


class MetaclassClassificationTaskExpander(TaskExpander):
    """MetaclassClassificationTaskExpander produces new tasks for each meta-class grouping of old classes.
    """
    def __call__(self, task, task_map, flatten_binary=False, force_remove_multi_label=False,
                 transform=None, target_transform=None):
        """Call function.

        Parameters
        ----------
        task : ClassificationTaskDataset
            old task
        task_map : list
            list of new tasks, where each task is provided as the following dict:
                {
                    "task_id": int,          # new task id
                    "task_name": str,        # new task name
                    "class_names": [str],    # NEW class names
                    "class_ids": [int],      # NEW class_ids
                    "label_map": {int: int}, # old class_id -> new class_id
                    "multi_label": int(0|1), # whether to force multilabel (optional)
                }
        flatten_binary : bool (default False)
            if True and label space is binary, flatten [0,1] new label to 1
        force_remove_multi_label : bool (default False)
            if True, force removing samples with multiple labels
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : task_id (int) -> ClassificationTaskDataset
        """
        assert isinstance(task_map, list)
        assert all(isinstance(task, dict) for task in task_map)
        assert all(all(x in task for x in ['task_id', 'task_name', 'label_map']) for task in task_map), 'task_map missing some required keys'

        new_tasks = {}
        for task_mapper in task_map:
            new_task_id = task_mapper['task_id']
            new_task_name = task_mapper['task_name']
            label_map = task_mapper['label_map']
            force_multi_label = task_mapper['multi_label'] if 'multi_label' in task_mapper else 0
            assert force_multi_label in [0,1]

            # get new label names if available
            label_names = None
            if 'class_names' in task_mapper and 'class_ids' in task_mapper:
                new_class_ids = task_mapper['class_ids']
                new_class_names = task_mapper['class_names']
                assert len(new_class_names) == len(new_class_ids)
                label_names = {i:n for i, n in zip(new_class_ids, new_class_names)}

            # create new task
            remapper = ClassificationTaskExpander()
            new_task = remapper(task,
                                label_map,
                                task_id=new_task_id,
                                task_name=new_task_name,
                                metadata=task.metadata,
                                label_names=label_names,
                                flatten_binary=flatten_binary,
                                force_multi_label=bool(force_multi_label),
                                force_remove_multi_label=force_remove_multi_label,
                                transform=transform,
                                target_transform=target_transform)
            new_tasks[new_task_id] = new_task

        return new_tasks


class BinaryClassificationTaskExpander(MetaclassClassificationTaskExpander):
    """BinaryClassificationTaskExpander produces new binary tasks for each attribute.
    """
    def __call__(self, task, labels=None, transform=None, target_transform=None):
        """Call function.

        Parameters
        ----------
        task : ClassificationTaskDataset
            old task
        labels : list
            list of labels to consider
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        dict : label (int) -> ClassificationTaskDataset
        """
        if labels is None:
            labels = task.possible_labels
        elif isinstance(labels, int):
            labels = [labels]
        assert isinstance(labels, list)

        task_map = []
        for label in labels:
            label_name = task.label_names_map[label] if task.possible_label_names is not None else str(label)

            # create new task
            binary_task = {}
            binary_task['task_id'] = label
            binary_task['task_name'] = label_name
            binary_task['label_map'] = { l: 1 if l == label else 0 for l in task.possible_labels }
            binary_task['class_ids'] = [1, 0]
            binary_task['class_names'] = [label_name, 'not {}'.format(label_name)]
            task_map.append(binary_task)

        return super(BinaryClassificationTaskExpander, self).__call__(task,
                                                                      task_map,
                                                                      flatten_binary=True,
                                                                      transform=transform,
                                                                      target_transform=target_transform)


class UnionClassificationTaskExpander(TaskExpander):
    """UnionClassificationTaskExpander combines multiple tasks into one task.

    Supports different label merging strategies:
        DISJOINT : do not merge; remap all old classes to new classes
        LABELS : merge common classes
        LABEL_NAMES : merge classes with common names

    Parameters
    ----------
    merge_mode : str
        one of the following merge modes:
            DISJOINT : do not merge; remap all old classes to new classes
            LABELS : merge common classes
            LABEL_NAMES : merge classes with common names
    merge_duplicate_images : bool (default True)
        if True, merge duplicate images otherwise do not
    """
    DISJOINT_MERGE = 'DISJOINT'
    MERGE_LABELS = 'LABELS'
    MERGE_LABEL_NAMES = 'LABEL_NAMES'
    POSSIBLE_MERGE_MODES = [DISJOINT_MERGE, MERGE_LABELS, MERGE_LABEL_NAMES]

    def __init__(self, merge_mode=DISJOINT_MERGE, merge_duplicate_images=True):
        assert merge_mode in UnionClassificationTaskExpander.POSSIBLE_MERGE_MODES
        self._merge_mode = merge_mode
        self._merge_duplicate_images = merge_duplicate_images

    def __call__(self, tasks, task_id=None, task_name=None, metadata=None,
                 transform=None, target_transform=None):
        """Call function.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset
        task_id : int, default None
            task id (simply used as metadata)
        task_name : str, default None
            task name (simply used as metadata)
        metadata : dict (default empty)
            extra arbitrary metadata
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        ClassificationTaskDataset
        """
        assert isinstance(tasks, collections.Iterable)

        # build label map depending on mode
        if self._merge_mode == UnionClassificationTaskExpander.DISJOINT_MERGE:
            label_map, label_names_map = self._disjoint_merge(tasks)

        elif self._merge_mode == UnionClassificationTaskExpander.MERGE_LABELS:
            label_map, label_names_map = self._merge_labels(tasks)

        elif self._merge_mode == UnionClassificationTaskExpander.MERGE_LABEL_NAMES:
            label_map, label_names_map = self._merge_label_names(tasks)

        # remap old labels to new labels
        new_images, new_labels = self._remap_tasks(tasks, label_map)

        # combine labels for the same image across datasets
        if self._merge_duplicate_images:
            new_images, new_labels = self._merge_duplicate_images(new_images, new_labels)

        # create new metadata
        new_metadata = {}
        for idx, t in enumerate(tasks):
            new_metadata[idx] = deepcopy(t.metadata)

        return ClassificationTaskDataset(new_images,
                                         new_labels,
                                         label_names=label_names_map,
                                         task_id=task_id,
                                         task_name=task_name,
                                         metadata=new_metadata if metadata is None else metadata,
                                         transform=transform,
                                         target_transform=target_transform)

    def _disjoint_merge(self, tasks):
        """Helper for disjoint merge.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset

        Returns
        -------
        label_map : dict
            map of task (index) -> old label -> new label
        label_names_map : dict
            map of new label -> new label name
        """
        label_map = {}
        label_names_map = {}

        label_counter = 0
        for idx, t in enumerate(tasks):
            label_map[idx] = {l : label_counter+i for i, l in enumerate(t.possible_labels)}
            if t.possible_label_names is not None:
                label_names_map.update({label_counter+i : n for i, n in enumerate(t.possible_label_names)})
            else:
                task_name = t.task_name if t.task_name is not None else str(idx)
                label_names_map.update({label_counter+i : '{}_{}'.format(task_name, l) for i, l in enumerate(t.possible_labels)})
            label_counter += len(t.possible_labels)

        return label_map, label_names_map

    def _merge_labels(self, tasks, intersection=False):
        """Helper for labels merge.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset
        intersection : bool (default False)
            if True, only return intersection

        Returns
        -------
        label_map : dict
            map of task (index) -> old label -> new label
        label_names_map : dict
            map of new label -> new label name
        """
        if intersection:
            common_labels = set.intersection(*[set(t.possible_labels) for t in tasks])
            if len(common_labels) == 0:
                return {}, {}

        label_map = {}
        label_names_map = {}

        label_counter = 0
        old_labels = {}
        for idx, t in enumerate(tasks):
            label_map[idx] = {}
            for i, l in enumerate(t.possible_labels):
                if intersection and l not in common_labels:
                    continue

                if l in old_labels:
                    new_label = old_labels[l]
                else:
                    new_label = label_counter
                    old_labels[l] = new_label
                    label_counter += 1
                label_map[idx][l] = new_label

                # merge label names by concatenating names
                if t.possible_label_names is not None:
                    if new_label not in label_names_map:
                        label_names_map[new_label] = t.possible_label_names[i]
                    else:
                        label_names_map[new_label] += ',' + t.possible_label_names[i]
                else:
                    task_name = t.task_name if t.task_name is not None else str(idx)
                    if new_label not in label_names_map:
                        label_names_map[new_label] = '{}_{}'.format(task_name, l)
                    else:
                        label_names_map[new_label] += ',{}_{}'.format(task_name, l)

        return label_map, label_names_map

    def _merge_label_names(self, tasks, intersection=False):
        """Helper for label names merge.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset
        intersection : bool (default False)
            if True, only return intersection

        Returns
        -------
        label_map : dict
            map of task (index) -> old label -> new label
        label_names_map : dict
            map of new label -> new label name
        """
        if intersection:
            common_label_names = set.intersection(*[set(t.possible_label_names) for t in tasks])
            if len(common_label_names) == 0:
                return {}, {}

        label_map = {}
        label_names_map = {}

        label_counter = 0
        old_labels = {}
        for idx, t in enumerate(tasks):
            assert t.possible_label_names is not None, idx
            label_map[idx] = {}
            for i, l in enumerate(t.possible_labels):
                label_name = t.possible_label_names[i]
                if intersection and label_name not in common_label_names:
                    continue

                if label_name in old_labels:
                    new_label = old_labels[label_name]
                else:
                    new_label = label_counter
                    old_labels[label_name] = new_label
                    label_counter += 1
                label_map[idx][l] = new_label
                label_names_map[new_label] = label_name

        return label_map, label_names_map

    def _remap_tasks(self, tasks, label_map):
        """Helper to remap task labels.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset
        label_map : dict
            map of task (index) -> old label -> new label

        Returns
        -------
        new_images : list
        new_labels : list
        """
        remapper = ClassificationTaskExpander()
        new_images = []
        new_labels = []
        for idx, t in enumerate(tasks):
            new_task = remapper(t, label_map[idx], label_names=None)
            new_images.extend([os.path.join(t.root, img) if isinstance(img, str) else img for img in new_task.images])
            new_labels.extend(new_task.labels)
        return new_images, new_labels

    def _merge_duplicate_images(self, images, labels):
        """Merge duplicate images.

        Parameters
        ----------
        images : list
        labels : list

        Returns
        -------
        new_images : list
        new_labels : list
        """
        new_data = {}
        for img, label in zip(images, labels):
            assert isinstance(img, str)
            if img not in new_data:
                new_data[img] = label
            elif isinstance(label, int):
                if isinstance(new_data[img], int):
                    new_data[img] = [new_data[img], label]
                else:
                    new_data[img].append(label)
            elif isinstance(label, list):
                if isinstance(new_data[img], int):
                    new_data[img] = [new_data[img], *label]
                else:
                    new_data[img].extend(label)
            else:
                ValueError()

        new_images = []
        new_labels = []
        for k,v in new_data.items():
            new_images.append(k)
            new_labels.append(v)

        return new_images, new_labels


class IntersectionClassificationTaskExpander(UnionClassificationTaskExpander):
    """IntersectionClassificationTaskExpander combines multiple tasks with common labels into one task.

    Supports different label merging strategies:
        LABELS : merge common classes
        LABEL_NAMES : merge classes with common names

    Parameters
    ----------
    merge_mode : str
        one of the following merge modes:
            LABELS : merge common classes
            LABEL_NAMES : merge classes with common names
    """
    MERGE_LABELS = 'LABELS'
    MERGE_LABEL_NAMES = 'LABEL_NAMES'
    POSSIBLE_MERGE_MODES = [MERGE_LABELS, MERGE_LABEL_NAMES]

    def __init__(self, merge_mode=MERGE_LABEL_NAMES):
        assert merge_mode in IntersectionClassificationTaskExpander.POSSIBLE_MERGE_MODES
        self._merge_mode = merge_mode

    def __call__(self, tasks, task_id=None, task_name=None, metadata=None,
                 transform=None, target_transform=None):
        """Call function.

        Parameters
        ----------
        tasks : list
            list of ClassificationTaskDataset
        task_id : int, default None
            task id (simply used as metadata)
        task_name : str, default None
            task name (simply used as metadata)
        metadata : dict (default empty)
            extra arbitrary metadata
        transform : callable (default None)
            Optional transform to be applied on a sample.
        target_transform : callable (default None)
            Optional transform to be applied on a label.

        Returns
        -------
        ClassificationTaskDataset
            or
        None if intersection is empty
        """
        assert isinstance(tasks, collections.Iterable)

        # build label map depending on mode
        if self._merge_mode == IntersectionClassificationTaskExpander.MERGE_LABELS:
            label_map, label_names_map = self._merge_labels(tasks, intersection=True)

        elif self._merge_mode == IntersectionClassificationTaskExpander.MERGE_LABEL_NAMES:
            label_map, label_names_map = self._merge_label_names(tasks, intersection=True)

        # if no overlap
        if len(label_map) == 0 and len(label_names_map) == 0:
            return None

        # remap old labels to new labels
        new_images, new_labels = self._remap_tasks(tasks, label_map)

        # combine labels for the same image across datasets
        new_images, new_labels = self._merge_duplicate_images(new_images, new_labels)

        # create new metadata
        new_metadata = {}
        for idx, t in enumerate(tasks):
            new_metadata[idx] = deepcopy(t.metadata)

        return ClassificationTaskDataset(new_images,
                                         new_labels,
                                         label_names=label_names_map,
                                         task_id=task_id,
                                         task_name=task_name,
                                         metadata=new_metadata if metadata is None else metadata,
                                         transform=transform,
                                         target_transform=target_transform)


class TrainTestExpander(UnionClassificationTaskExpander):
    """TrainTestExpander ensures targets from both train and test partitions are the same.

    This is basically IntersectionClassificationTaskExpander but does not merge into one dataset.

    Supports different label merging strategies:
        LABELS : merge common classes
        LABEL_NAMES : merge classes with common names

    Parameters
    ----------
    merge_mode : str
        one of the following merge modes:
            LABELS : merge common classes
            LABEL_NAMES : merge classes with common names
    """
    MERGE_LABELS = 'LABELS'
    MERGE_LABEL_NAMES = 'LABEL_NAMES'
    POSSIBLE_MERGE_MODES = [MERGE_LABELS, MERGE_LABEL_NAMES]

    def __init__(self, merge_mode=MERGE_LABEL_NAMES):
        assert merge_mode in TrainTestExpander.POSSIBLE_MERGE_MODES
        self._merge_mode = merge_mode

    def __call__(self, *tasks):
        """Call function.

        Parameters
        ----------
        *tasks : list (positional arguments)
            list of ClassificationTaskDataset

        Returns
        -------
        list of ClassificationTaskDataset
        """
        assert isinstance(tasks, collections.Iterable)

        # build label map depending on mode
        if self._merge_mode == TrainTestExpander.MERGE_LABELS:
            label_map, label_names_map = self._merge_labels(tasks, intersection=True)

        elif self._merge_mode == TrainTestExpander.MERGE_LABEL_NAMES:
            label_map, label_names_map = self._merge_label_names(tasks, intersection=True)

        # if no overlap
        if len(label_map) == 0 and len(label_names_map) == 0:
            raise ValueError('tasks provided do not share any common labels!')

        # build new tasks with consistent label map
        new_tasks = []
        remapper = ClassificationTaskExpander()
        for idx, t in enumerate(tasks):
            new_task = remapper(t, label_map[idx], label_names=label_names_map,
                                task_id=t.task_id, task_name=t.task_name, metadata=t.metadata,
                                transform=t.transform, target_transform=t.target_transform)
            new_tasks.append(new_task)

        return new_tasks
