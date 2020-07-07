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
  echo "Usage: ./download_inat2018.sh DATA_ROOT"
  exit 1
fi

DATA_DIR=$1
TASK2VEC_REPO="./"

mkdir -p "$DATA_DIR"/inat2018
cd "$DATA_DIR"/inat2018|| exit 1

# For alternative ways of downloading the files check https://github.com/visipedia/inat_comp/tree/master/2018
FILES="train_val2018.tar.gz train2018.json.tar.gz val2018.json.tar.gz test2018.tar.gz test2018.json.tar.gz"
for FILE in $FILES ; do
  echo "Downloading $FILE..."
  wget https://storage.googleapis.com/inat_data_2018_us/$FILE
  echo "Extracting $FILE..."
tar -xzf $FILE
done
echo "Downloading unobfuscated category names"
wget http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/categories.json.tar.gz
tar -xzf categories.json.tar.gz

# Auxiliary files needed to generate the task list in the paper:
cp $TASK2VEC_REPO/support_files/inat2018/*.json ./