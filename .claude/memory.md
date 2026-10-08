# Servers

The repo is cloned on three servers, A, B and C. Work out which one a session runs on before running anything on a GPU or using a machine path. The working directory usually tells; the GPU names from `nvidia-smi --query-gpu=name --format=csv,noheader` settle it.

| Server | Working directory | GPUs |
| --- | --- | --- |
| A | `/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD` | 2× NVIDIA RTX 6000 Ada Generation |
| B | `/home/<user>/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD` | 4× NVIDIA H200 |
| C | `/home/users/<user>/projects/OOD_PanSeg_3D/P3Former_OOD` | none on the login node, which has no `nvidia-smi` but has `/opt/pbs`; Tesla V100-SXM2-32GB inside a PBS job |

If nothing matches, this is a new machine: ask the user, then add it here without its address.

`environment.md` and `machine-resources.md` describe A; on B and C, the sections below take their place. Results can differ slightly between servers, since the GPUs differ and B runs a newer torch, so compare numbers made on the same server.

The repo is public, so login addresses (hosts, IPs, users, jump hosts) stay out of it. Claude's auto-memory on A has them; elsewhere, ask the user.

## A: the main workstation

- Interpreter, hardware, disk and long jobs: see `environment.md` and `machine-resources.md`.
- A holds every dataset, checkpoint and result. `tools/copy_to_server.sh` copies them to B and C.
- In Claude Code's shell, `LD_LIBRARY_PATH` breaks `ssh`: run `env -u LD_LIBRARY_PATH ssh ...`, and the same for `rsync`.

## B: a 4× H200 server

- Interpreter: `~/anaconda3/envs/p3former/bin/python`. It has torch 2.0.1+cu118 instead of the pinned 1.10.1+cu111, which has no kernels for the H200 (sm_90); mmcv, mmdet, mmdet3d and mmengine are the pinned versions. On 2026-10-08 it imported and saw the GPUs; no train or eval run with it has been checked yet.
- Clone: `~/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD`. The data copied from A is in `~/khoadv/projects/OOD_PanSeg_3D/store`, symlinked into the clone.
- Disk: `/home` has ~2.2 TB free and `/` only ~8 GB. There is no `/mnt/sandisk`; put large outputs in the store.
- The login account is shared with other projects: keep everything under `~/khoadv/`.
- GPU jobs run as on A: check `nvidia-smi`, pick a GPU with `CUDA_VISIBLE_DEVICES`, and start long jobs with `nohup setsid`.
- Copied from A on 2026-10-08: the core tier of `copy_to_server.sh` (DSO data and pkls, the DSO checkpoint, small result logs). The SemanticKITTI and analysis tiers are not there. Copies run at ~2.8 MB/s through a relay.

## C: a PBS cluster with V100s

- The login node has no GPU. GPU work runs only as PBS jobs submitted with `qsub`, never with `CUDA_VISIBLE_DEVICES` or `nohup` on the login node. A job keeps running after the session ends; `qstat -u $USER` lists it. The PBS commands are in `/opt/pbs/bin`, which a non-login shell may not have on `PATH`.
- The user can submit to `v100q` and `aimcq`, not to `a100q` or `h100q`. The nodes: three with 8× V100-SXM2-32GB and one with 4, plus an 8× A100 node that was offline on 2026-10-08. Walltime is at most 100 h. The V100 is sm_70, which the pinned torch supports.
- The user's job header:

  ```bash
  #!/bin/bash
  #PBS -N <job name>
  #PBS -q v100q
  #PBS -l select=1:ncpus=40:ngpus=1:mem=256gb
  #PBS -l walltime=99:00:00
  #PBS -P 20260103
  #PBS -j oe
  #PBS -o <absolute path of the log>
  cd "$PBS_O_WORKDIR" || exit $?
  ```

  `~/Trash/gpu_check.pbs` is a 5-minute job that checks the GPU.
- Interpreter: `~/miniconda3/envs/p3former/bin/python`, with the pinned stack. On 2026-10-08 it imported on the login node; no GPU job has used it yet.
- Clone: `~/projects/OOD_PanSeg_3D/P3Former_OOD`. The data copied from A is in `~/projects/OOD_PanSeg_3D/store`, symlinked into the clone.
- Disk: `/home/users` is NFS shared with the compute nodes, with ~129 TB free. It compresses, so `du` shows the 128 GB store as ~79 GB; `du --apparent-size` matches A. `/scratch` (BeeGFS) exists only on the compute nodes. There is no `/mnt/sandisk`; put large outputs in the store.
- Copied from A on 2026-10-08: the core tier, verified identical to A.
