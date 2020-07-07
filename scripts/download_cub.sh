#!/bin/sh

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

if [ $# -nq 1 ]; then
  echo "Usage: ./download_cub.sh DATA_ROOT"
  exit 1
fi

DATA_DIR=$1
TASK2VEC_REPO="./"


mkdir -p "$DATA_DIR"/cub
cd "$DATA_DIR"/cub || exit 1

wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar -xzf images.tgz
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz
tar -xzf lists.tgz
mv lists/* CUB_200_2011/
cp $TASK2VEC_REPO/support_files/cub/*.json "$DATA_DIR"/cub/CUB_200_2011