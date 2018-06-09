conda create --name jztorch python=3.5

conda install pytorch torchvision -c pytorch

conda install -c conda-forge tqdm

conda install pathlib

conda install pretrainedmodels

#!/bin/bash
dir1="/srv/jing/subject_find/kaggle_furniture/dat/train_copy/chunk5/"
dir2="/srv/jing/subject_find/kaggle_furniture/dat/train/"
s1="/*"
backslash="/"
array=($(ls "/srv/jing/subject_find/kaggle_furniture/dat/chunk0/"  | cut -d"-" -f1))
arraylength=${#array[@]}
for (( i=1; i<${arraylength}+1; i++ ));
do
  mv $dir1${array[$i-1]}$s1  $dir2${array[$i-1]}$backslash
done

