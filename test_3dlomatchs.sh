# /usr/bin/bash

for i in `ls ./3DLoMatch/categories`;
do
filename=${i%.*}
python -m scripts.test_3dmatch --threed_match_dir /root/datasets/3dmatch/ --weights ResUNetBN2C-feat32-3dmatch-v0.05.pth --test_overlap_file 3DLoMatch/categories/$i > 3DLoMatch/logs/$filename.txt
done
