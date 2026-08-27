set -ex

for i in $(seq 1 17); do
  CUDA_VISIBLE_DEVICES=1 python test.py \
    configs/p3former/hier/p3former_2xb1_3x_dso_ood_hier_b$i.py \
    work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \
    --work-dir work_dirs/p3former_2xb1_3x_dso_ood_hier/b$i
done



python tools/summarize_hierarchy_ablation.py --family group --exclude odin --plot top4_group.png work_dirs/p3former_2xb1_3x_dso_ood_hier/b*/*/*.log