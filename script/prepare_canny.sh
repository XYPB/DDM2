#! /bin/bash
set +x

# python data/prepare_canny.py --raw_data experiments/hardi150_denoise_230426_211632_baseline/results/hardi150_denoised.nii.gz
# python data/prepare_canny.py --raw_data experiments/hardi150_denoise_230430_033002_ddim/results/hardi150_denoised.nii.gz
# python data/prepare_canny.py --raw_data experiments/hardi150_denoise_230419_142729_noise/results/hardi150_denoised.nii.gz
# python data/prepare_canny.py --raw_data experiments/hardi150_denoise_230430_115119_control/results/hardi150_denoised.nii.gz
# python data/prepare_canny.py --raw_data experiments/hardi150_p2s/denoised_StanfordHardi_p2s_mlp.nii.gz

python data/prepare_canny.py --raw_data experiments/s3sh_denoise_230426_233927_baseline/results/s3sh_denoised.nii.gz
python data/prepare_canny.py --raw_data experiments/s3sh_denoise_230430_045617_ddim/results/s3sh_denoised.nii.gz
python data/prepare_canny.py --raw_data experiments/s3sh_denoise_230419_170618_noise/results/s3sh_denoised.nii.gz
python data/prepare_canny.py --raw_data experiments/s3sh_denoise_230430_224918_control/results/s3sh_denoised.nii.gz
python data/prepare_canny.py --raw_data experiments/s3sh_p2s/denoised_sherbrooke_3shell_p2s_mlp.nii.gz