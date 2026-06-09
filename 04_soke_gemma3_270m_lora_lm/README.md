# 04 SOKE Gemma 3 270M LoRA LM

This bundle trains a single-head causal LM over SOKE-style sign motion tokens.
It keeps SOKE's data formulation:

- source aliases are `how2sign`, `csl`, and `phoenix`
- prompt rows use SOKE's official instruction dictionaries from
  `ref/SOKE/prepare/instructions/`
- motion targets use SOKE token namespaces: `<motion_id_*>`, `<hand_id_*>`,
  and `<rhand_id_*>`
- the causal target is the SOKE flattened triplet form:
  `<motion_id_b> <hand_id_l> <rhand_id_r> ...`

It intentionally does not use SOKE's multi-head mBART decoder. Gemma predicts one
causal token stream.

## Files

- `04_soke_gemma3_270m_lora_lm_colab.ipynb`: Colab training notebook.
- `04_soke_gemma3_270m_lora_lm_inspect.ipynb`: local/Colab dataset inspection notebook.
- `scripts/build_soke_motion_codes.py`: local SOKE VQ-VAE code builder.
- `scripts/train_gemma_soke_lora.py`: Gemma 3 270M LoRA trainer.
- `scripts/inspect_soke_gemma_dataset.py`: random sample and tokenization audit CLI.
- `scripts/soke_gemma_data.py`: shared dataset/instruction utilities.
- `outputs/soke_motion_codes/`: locally built SOKE-code JSONL rows.

## Data Flow

1. Build SOKE motion-code JSONL locally from the existing new-data SOKE20
   manifests:

   ```bash
   LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-} \
   python new_data/04_soke_gemma3_270m_lora_lm/scripts/build_soke_motion_codes.py \
     --manifest-root new_data/02_soke20fps_VQVAE_tokenizer/outputs/preprocess_soke20 \
     --output-root new_data/04_soke_gemma3_270m_lora_lm/outputs/soke_motion_codes \
     --splits train val test
   ```

   The builder resolves How2Sign/Neural captions from the existing
   `pkl_to_rgb_front_index.csv` and original How2Sign sentence-level CSVs, CSL
   captions from `csl_clean.*`, and Phoenix captions from SOKE-compatible
   `phoenix14t.*` files or extra text indexes if supplied. Rows without
   captions are skipped and counted in `build_summary.csv`.

2. Inspect the generated rows:

   ```bash
   python new_data/04_soke_gemma3_270m_lora_lm/scripts/inspect_soke_gemma_dataset.py \
     --code-root new_data/04_soke_gemma3_270m_lora_lm/outputs/soke_motion_codes \
     --split train --stage lm_pretrain --n 5
   ```

3. Sync this directory to Colab and run the Colab notebook. The notebook copies
   JSONL data from Drive to `/content`, trains locally, and syncs logs/adapters
   back to Drive.

## Model

The default base model is `google/gemma-3-270m`. Hugging Face access still
requires accepting Google's model license on the account used by Colab. Override
with `SOKE_GEMMA_BASE_MODEL` if needed.

In Colab, add an `HF_TOKEN` secret after accepting the Gemma license. The
notebook copies that secret into the environment so the training script can load
the gated model. The pre-training inspect cell does not require tokenizer/model
access by default; set `SOKE_GEMMA_INSPECT_TOKENIZER_AUDIT=1` if you also want
the optional tokenizer audit.

The dependency cell installs `torchao>=0.16.0` because current PEFT checks the
installed `torchao` package during LoRA injection. Colab images may include an
older `torchao` such as `0.10.0`, which causes PEFT to fail before training
starts.

Training uses PEFT LoRA plus PEFT trainable token rows for the newly added SOKE
motion-token embeddings. Defaults:

- LoRA LR: `2e-4`
- motion-token embedding LR: `2e-5`
- LoRA rank/alpha: `r=64`, `alpha=128`
- LoRA targets: `auto`, which resolves Gemma's attention projections
  `q_proj,k_proj,v_proj,o_proj` and MLP projections
  `gate_proj,up_proj,down_proj` across all transformer layers.

## Training Strategy

The Colab notebook exposes the batch knobs at the top:

- `HARDWARE_BATCH_SIZE`: per-step/per-GPU microbatch, change this for GPU memory.
- `GRAD_ACCUM`: gradient accumulation steps.
- `VIRTUAL_BATCH_SIZE = HARDWARE_BATCH_SIZE * GRAD_ACCUM`.
- `GRADIENT_CHECKPOINTING`: default `1`, reduces activation memory for the large
  default microbatch.
- `ATTN_IMPLEMENTATION`: default `sdpa`; set `auto`, `eager`, or
  `flash_attention_2` if the runtime/model stack needs a different attention
  backend.

The default is hardware batch `64` and grad accumulation `2`, so the virtual
batch is `128`. A hardware batch of `128` is not viable with full Gemma vocab
cross-entropy in this setup because the logits/loss tensor scales with
`batch * sequence_length * 262k_vocab`. Reduce hardware batch on smaller GPUs
and increase grad accumulation if you need to keep the virtual batch stable.

Notebook subprocess calls stream their output and print the last lines again on
failure, so Colab errors should show the underlying training traceback instead
of only `CalledProcessError`.

The notebook currently defaults to the instruction stage because the pretrain
run has already produced Drive checkpoints:

- `lm_instruct`: 150 epochs

By default, `lm_instruct` initializes from `lm_pretrain/last_adapter` when that
adapter exists. Set `SOKE_GEMMA_INSTRUCT_INIT_ADAPTER_NAME=best_adapter` if you
want to start instruction tuning from the best validation pretrain checkpoint
instead. Set `SOKE_GEMMA_RUN_STAGES=lm_pretrain,lm_instruct` to run both stages
from scratch again.

Validation defaults:

- teacher-forced val loss/token accuracy at epoch 1, every 5 epochs, and final
  epoch
- sampled generated motion-code validation at the same cadence
- motion-code sample fraction `0.10`, capped at `256` validation rows

The in-training motion validation is intentionally lightweight for Colab. It
generates SOKE token streams from text-to-motion prompts and reports valid
triplet ratio, generated/target length ratio, and exact body/left-hand/right-hand
code accuracy. Physical SMPL-X MPJPE/PA/DTW metrics still require the SOKE
decoder and SMPL-X body models and should be run as a heavier offline evaluation
when needed.

Every epoch saves:

- root `last_adapter`
- root `best_adapter`
- optimizer/scheduler train state
- `history.csv`, event logs, train config, and tokenization audit
- an `epoch_saves/epoch_XXXX/` snapshot containing both `last_adapter` and
  `best_adapter`

Retention defaults keep the last 5 epoch snapshots locally and the last 5 epoch
snapshots on Drive. Drive sync uses retrying `rsync` with `--delete` inside the
isolated run directory so old snapshots are pruned instead of accumulating.

## Local Build Status

The local SOKE-code dataset was built at:

```text
new_data/04_soke_gemma3_270m_lora_lm/outputs/soke_motion_codes
```

Rows written:

- train: 39,424 rows (`csl` 12,018, `how2sign` 22,054, `phoenix` 5,352)
- val: 2,326 rows (`csl` 748, `how2sign` 1,210, `phoenix` 368)
- test: 2,880 rows (`csl` 837, `how2sign` 1,620, `phoenix` 423)

Rows skipped:

- train: 0
- val: 0
- test: 0

No SOKE VQ-VAE feature/model extraction failures occurred. The generated JSONLs
cover every accepted row in the current preprocessed manifests.

SOKE's LM renderer uses code-token length for `<Frame_Placeholder>` in this
stage. This bundle mirrors that behavior for instruction rows while preserving
the real 20 fps frame count in JSONL metadata as `num_frames_soke`.
