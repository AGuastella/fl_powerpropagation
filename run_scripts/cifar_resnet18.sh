#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:rtx2080:1

cd /nfs-share/ag2411/project/fl_powerpropagation
poetry run python -m project.main --config-name=cluster_cifar_resnet18 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.01 fed.num_rounds=700 dataset.lda_alpha=1000
