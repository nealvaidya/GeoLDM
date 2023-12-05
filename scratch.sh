python eval_analyze.py --model_path outputs/drugs_latent2 --n_samples 1_000

docker run -it --rm --gpus all --network host --shm-size=1g -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.10-py3

pip install wandb imageio rdkit matplotlib==3.5.2
git config --global --add safe.directory /workspace/GeoLDM



python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 100 --diffusion_loss_type l2 --batch_size 4 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9_100steps --resume outputs/geoldm_qm9_100steps_resume

CUDA_VISIBLE_DEVICES=2 python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 10 --diffusion_loss_type l2 --batch_size 128 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9_10steps