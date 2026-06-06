# 02 SOKE20FPS NB51 Tokenizer

Colab-ready tokenizer workspace for training a Notebook-51-style joint-token VQ-VAE on the new SOKE_DATA datasets:

- `Neural-Sign-Actors` is treated as the How2Sign source.
- `CSL-Daily-Fittings` is treated as CSL.
- `PHOENIX` is treated as Phoenix.

The default model matches the saved Notebook 51 `t08_s32_cd1024` row:

- `64` input frames per window.
- `8 x 32 = 256` VQ codes per 64-frame window.
- `num_bins=128`.
- `d_model=256`.
- `code_dim=1024`.
- `code_num=1024`.
- `num_heads=8`, `num_kv_heads=4`.
- `spatial_blocks=2`, `temporal_blocks=2`.

Preprocessing follows SOKE-style filtering/downsampling before the Notebook 51 canonical target:

- Downsample frame folders to `20 fps` by deterministic frame selection.
- Drop SOKE's fixed bad How2Sign IDs.
- Drop How2Sign/Neural clips with duration `>= 30s`.
- Drop clips with fewer than `4` source frames.
- Then drop clips with fewer than `64` frames after 20 fps downsampling, because Notebook 51 cannot train a 64-frame window from them.

## Files

- `02_soke20fps_nb51_tokenizer_colab.ipynb`: Colab orchestration notebook.
- `combined_clip_index.csv`: prebuilt clip index from notebook 01, used to avoid a fresh folder scan.
- `scripts/unpack_recursive_zips.py`: Colab-safe SOKE_DATA extractor for `../DATA/ZIPS`.
- `scripts/soke20_preprocess.py`: builds 20 fps cache NPZs and train/val/test manifests.
- `scripts/train_nb51_soke20.py`: trains the VQ-VAE and writes history/checkpoints.
- `scripts/eval_nb51_soke20_metrics.py`: computes per-dataset full-clip MPJPE/JPE, PA-MPJPE, DTW-MPJPE, and DTW-PA-MPJPE validation summaries.
- `scripts/joint_token_rmsnorm_gqa_vqvae.py`: Notebook 51 model definition.
- `scripts/smplx_motiongpt_preprocess.py`: canonicalization used by Notebook 51.

## Local Smoke Test

```bash
python scripts/soke20_preprocess.py \
  --data-root ../../DATA/SOKE_DATA \
  --index-csv ../01_csl_phoenix_neural_sign_actors_exploration_and_stats/combined_clip_index.csv \
  --out-root smoke_outputs \
  --target-fps 20 \
  --window-size 64 \
  --max-clips-per-split 2 \
  --force

python scripts/train_nb51_soke20.py \
  --preprocess-root smoke_outputs \
  --run-root smoke_train \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --rebuild-quantizer

python scripts/eval_nb51_soke20_metrics.py \
  --preprocess-root smoke_outputs \
  --run-root smoke_train \
  --body-model-root ../../body_models \
  --max-clips-per-dataset 1
```

## Colab Notes

The Drive bundle is intended to live at:

```text
/content/drive/MyDrive/folder/COLAB/Tokenizer
```

The data zips should live one directory higher:

```text
/content/drive/MyDrive/folder/COLAB/DATA/ZIPS
```

Do not store the extracted dataset folders in `COLAB/DATA`; that creates a huge Drive sync burden. The notebook treats `COLAB/DATA` as zip storage only by default. It extracts the zips into the Colab VM at `/content/SOKE_DATA` when `SOKE20_AUTO_EXTRACT_DATA=1`. The extractor skips the harmless `/` root entries present in the CSL/Neural ZIPs so Colab's `unzip` backend does not fail with exit code `2`.

By default on Colab, training outputs are written locally to `/content/soke20_nb51_tokenizer_outputs` and the final cell robustly syncs checkpoints, logs, plots, summaries, and `final_save_status.json` to `/content/drive/MyDrive/folder/COLAB/Tokenizer/outputs`. The final sync copies the selected run folder first, then preprocess manifests without `cache_soke20/`, so checkpoints reach Drive before large cache data.

Set these in the notebook if needed:

- `SOKE_DATA_ROOT`: path to the extracted data root containing `CSL-Daily-Fittings`, `Neural-Sign-Actors`, and `phoenix_poses`.
- `SOKE20_DATA_ZIP_ROOT`: path to the folder containing the three top-level zips.
- `SOKE20_DATA_EXTRACT_ROOT`: extraction destination; defaults to `/content/SOKE_DATA` on Colab.
- `SOKE20_DATA_EXTRACT_FORCE=1`: remove and rebuild the runtime extraction folders. Use this after an interrupted/failed extract leaves partial `/content/SOKE_DATA` contents.
- `SOKE20_ALLOW_DRIVE_EXTRACTED_DATA=1`: opt-in only, allows using extracted dataset folders on Drive. Leave unset for normal Colab runs.
- `SOKE20_OUTPUT_ROOT`: override output folder. Leave unset for local Colab training outputs plus final Drive sync.
- `SOKE20_RUN_NAME`: override the training run folder name. By default it includes `code_dim`, `code_num`, hardware batch, and grad accumulation, for example `t08_s32_cd1024_cn1024_b32_ga4_soke20`.
- `SOKE20_RESUME_TRAINING=0`: starts from scratch inside the selected run folder. Use this, or choose a new `SOKE20_RUN_NAME`, when changing architecture settings like `SOKE20_CODE_NUM`.
- `SOKE20_INDEX_CSV`: optional prebuilt combined clip index.
- `SMPLX_MODEL_ROOT`: path to a folder containing `smplx/SMPLX_NEUTRAL.npz` if the notebook cannot auto-detect it.
- `SOKE20_VALIDATION_METRIC_MAX_CLIPS_PER_DATASET`: set to a small value for a metric smoke test; default `0` evaluates every validation clip.
- `SOKE20_SYNC_TO_DRIVE=0`: disables the final robust Drive sync.
- `SOKE20_KEEP_RUNTIME=1`: disables the final runtime unassign cell.
