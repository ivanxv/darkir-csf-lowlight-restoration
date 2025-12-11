# DarkIR-CSF: Cross-Scale Fusion Experiments for Low-Light Image Restoration

This repo explores a lightweight Cross-Scale Fusion (CSF) block on top of the DarkIR architecture for low-light, noisy, and blurry image restoration. The goal is to test CSF in a data-scarce setting and compare against the DarkIR baseline.

## Important Note (Data and Results)
- All experiments use a reduced LOLBlur subset: **1620 train**, **600 val**, **540 test** pairs.
- CSF improved validation PSNR/L1; on the held-out test set the vanilla DarkIR generalized slightly better.
- Results are exploratory; no SOTA claim.

## Project Summary
- Re-implemented DarkIR and trained on the LOLBlur subset.
- Added a CSF block on encoder-decoder skips.
- Compared DarkIR vs. DarkIR + CSF on PSNR, SSIM, LPIPS, params, and FLOPs.
- Aim: hands-on model design/training/evaluation with limited data plus a portfolio-ready demo.

## Repo Layout (abridged)
```
DarkIR/
  train.py, testing.py, inference.py, inference_video.py
  archs/, losses/, utils/
  options/
    train/ LOLBlur.yml, LOLBlur_csf.yml
    test/  LOLBlur.yml, LOLBlur_CSF.yml
  data/datasets/LOLBlur/ (train, train_val, test)   # place data here
  models/ DarkIR_lolblur_original_best.pt, DarkIR_lolblur_best.pt, ...
```
`options/train` and `options/test` hold paths and hyperparameters.

## Dataset
- LOLBlur pairs: low-light + blur/noise input, clean target.
- Split used: train 1620, val 600, test 540 pairs. Test is held out for final reporting.
- Data is not stored in the repo; download from the official source and place under `data/datasets/LOLBlur/`.

## Method
**DarkIR (baseline)**  
- Encoder-decoder with depth-wise + dilated convolutions.  
- ~3.32M params, 7.25 GMac (@3x256x256).

**DarkIR + CSF**  
- Skip fusion with a 1x1 conv MLP-style gate between encoder/decoder features.  
- ~3.38M params, 7.67 GMac (@3x256x256); small overhead.

## Why CSF?
- Baseline skip connections just add encoder and decoder features; CSF learns a per-channel gate to blend them, helping when blur/noise makes shallow skips noisy.
- In low-data training, the gate reduces over-reliance on the skip and nudges the decoder to use deeper context.
- Cost is small (+0.06M params, +0.4 GMac @3x256x256) for a modest validation gain (+0.3 dB PSNR, lower L1); on the held-out test set, performance is similar to baseline.

## Training Setup
- Patch 256x256 random crops, batch 4.
- AdamW (lr 1e-4, weight_decay 1e-4, betas 0.9/0.99), cosine schedule, eta_min 1e-6.
- Grad clip 5.0; loss: Charbonnier.
- Augment: horizontal/vertical flip.
- 100 epochs; mixed precision optional if CUDA is available.
- Hardware: local GPU or Colab T4; subset used due to VRAM limits.

## Train
```
# Baseline
python train.py -p ./options/train/LOLBlur.yml

# CSF variant
python train.py -p ./options/train/LOLBlur_csf.yml   # use_csf: true
```
Adjust paths/model names in `options/train/*` for your setup.

## Test
```
# CSF on LOLBlur
python testing.py -p ./options/test/LOLBlur_CSF.yml

# Baseline (set use_csf false and correct checkpoint path)
python testing.py -p ./options/test/LOLBlur.yml
```
- Metrics print to console.
- First 8 samples save to `save.results_dir` (default `./images/results_test`).
- Unpaired: `python testing_unpaired.py -p ./options/test/RealBlur_Night.yml`

## Inference (saves images)
```
python inference.py -p ./options/inference/LOLBlur.yml -i ./path/to/images
# outputs: ./images/results

python inference_video.py -p ./options/inference_video/Baseline.yml -i /path/to/video.mp4
# outputs: ./videos/results
```
Match checkpoint path and `use_csf` flag to the model you trained.

## Results (summary)
- Validation (600 pairs):
  - DarkIR: PSNR ~23.97, SSIM ~0.808, L1 ~0.0540 (epoch ~60)
  - DarkIR + CSF: PSNR ~24.32, SSIM ~0.797, L1 ~0.0478 (epoch ~20)
- Test (540 pairs):
  - DarkIR: PSNR 22.25, SSIM 0.6928, LPIPS 0.3858
  - DarkIR + CSF: PSNR 22.70, SSIM 0.7185, LPIPS 0.3641
- Takeaway: CSF helped on val; vanilla DarkIR edged ahead on test. More data/regularization/CSF design may close the gap.

## Qualitative Examples
- Example input/gt/output triplets are in `images/results_test` (or your `results_dir`). Below are CSF outputs.

| Input | Ground Truth | CSF Output |
| --- | --- | --- |
| ![](images/results_test/00000_input.png) | ![](images/results_test/00000_gt.png) | ![](images/results_test/00000_output.png) |
| ![](images/results_test/00001_input.png) | ![](images/results_test/00001_gt.png) | ![](images/results_test/00001_output.png) |
| ![](images/results_test/00002_input.png) | ![](images/results_test/00002_gt.png) | ![](images/results_test/00002_output.png) |

## Limitations and Next Steps
- Small subset and single dataset (LOLBlur).
- Future work: full LOLBlur training, other LLIE datasets, extra regularization (perceptual/TV), additional CSF variants, multi-task training.
