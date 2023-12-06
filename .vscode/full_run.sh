torchrun --nnodes=1 --nproc-per-node=8 distill_qm9.py \
    --teacher_model "outputs/qm9_latent2" \
    --no_wandb \
    --batch_size 128 \
    --num_workers 2 \
    --n_report_steps 10