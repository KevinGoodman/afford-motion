# EXP_NAME=$1
EXP_NAME=afford-trumans-adm


python train.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=500 \
            task=contact_gen \
            task.train.batch_size=512 \
            task.train.num_workers=9 \
            task.train.max_steps=10000 \
            task.train.log_every_step=10 \
            task.train.save_every_step=200 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            task.dataset.sets=['TRUMANS'] \
            model=cdm \
            model.arch=Perceiver
            