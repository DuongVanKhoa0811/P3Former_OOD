
set -ex


# PORT=29511 bash dist_train.sh configs/p3former/p3former_4xb1_3x_semantickitti.py 4
# CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_dso.py


# LD_LIBRARY_PATH= /usr/bin/rsync -avP -e /usr/bin/ssh /mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/SemanticKITTI/ khoa@172.31.124.78:/home/khoa/projects/OOD_PanSeg_3D/P3Former_OOD/data/semantickitti


# LD_LIBRARY_PATH= /usr/bin/rsync -avP -e /usr/bin/ssh /mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/SemanticKITTI/ khoadv@172.31.115.242:/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD/data/semantickitti/


# LD_LIBRARY_PATH= /usr/bin/rsync -avP -e /usr/bin/ssh khoa@172.31.124.78:/home/khoa/projects/OOD_PanSeg_3D/P3Former_OOD/work_dirs/p3former_4xb1_3x_semantickitti/ /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD/work_dirs/p3former_4xb1_3x_semantickitti/
