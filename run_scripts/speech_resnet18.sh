#!/bin/bash
#SBATCH -w ngongotaha
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1

cd /nfs-share/ag2411/project/fl_powerpropagation

# poetry run python -m project.main --config-name=cluster_speech_resnet18 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 

# poetry run python -m project.task.speech_resnet18.dataset_preparation --config-name=cluster_speech_resnet18 dataset.lda_alpha=0.1


# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_ZERO task.alpha=1.0 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_ZERO task.alpha=1.0 task.sparsity=0.95 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_ZERO task.alpha=1.0 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_ZERO task.alpha=1.0 task.sparsity=0.995 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 strategy=fedavgNZ
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_ZERO task.alpha=1.0 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 strategy=fedavgNZ
# strategy=fedavgNZ


# model_and_data: SPEECH_RESNET18
# model_and_data: SPEECH_PP
# model_and_data: SPEECH_PPSWAT
# model_and_data: SPEECH_ZERO


# resnet18 + topk
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.0 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.95 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.995 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_RESNET18 task.alpha=1.0 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 

# SparseFedPP
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.95 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.995 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PPSWAT task.alpha=1.25 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=1.0 fed.num_rounds=500 


# Powerprop + topk
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=5378 task.model_and_data=SPEECH_PP task.alpha=1.25 task.sparsity=0.9 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=5378 task.model_and_data=SPEECH_PP task.alpha=1.25 task.sparsity=0.95 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PP task.alpha=1.25 task.sparsity=0.99 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
# poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PP task.alpha=1.25 task.sparsity=0.995 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
poetry run python -m project.main --config-name=cluster_speech_resnet18 fed.seed=9421 task.model_and_data=SPEECH_PP task.alpha=1.25 task.sparsity=0.999 task.mask=0.0 task.fit_config.dataloader_config.batch_size=16 task.fit_config.run_config.learning_rate=0.1 task.fit_config.run_config.final_learning_rate=0.001 dataset.lda_alpha=0.1 fed.num_rounds=500 
