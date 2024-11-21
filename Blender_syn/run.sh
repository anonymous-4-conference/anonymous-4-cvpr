#!/bin/bash

cd /data/dataset/simulation_blender/4th_batch/slices/
rm -r *
cd /home/yishun/projectcode/paper3/simulation/blender/dataset/4th_batch/Liquor_models/
rm *


cd /home/yishun/projectcode/paper3/simulation/blender/codes

set -e
/home/yishun/blender/blender -b -P sample.py --python-use-system-env

if [ -z $(type -t conda) ] || [ "$(type -t conda)" != "function" ]; then
    source /yishun/anaconda3/etc/profile.d/conda.sh
fi
conda activate tomopy
python converter_fill_shape.py

#conda activate base
#python projection.py