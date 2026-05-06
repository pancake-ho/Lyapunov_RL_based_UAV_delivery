# Seraph Short HRL Train Smoke

This note is for validating that the HRL two-timescale PPO code can run a short GPU job on Seraph and leave reproducible logs, checkpoints, and sanity-check output. It is not for long training, reward coefficient tuning, or baseline comparison.

## 0. One-time setup

Keep runtime files under `/data/$USER`.

```bash
export DATA_ROOT=/data/$USER
export PROJECT_DIR=$DATA_ROOT/research
export CONDA_ENV=uav
export RUN_ROOT=$DATA_ROOT/hrl_uav
export LOG_DIR=$RUN_ROOT/logs
export CHECKPOINT_DIR=$RUN_ROOT/checkpoints
export OUTPUT_DIR=$RUN_ROOT/outputs

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR"
cd "$PROJECT_DIR"
```

If the repository is not already under `$PROJECT_DIR`, copy or clone it there before running the commands below.

## 1. Interactive GPU smoke with srun

Use this first to verify the conda environment, CUDA visibility, imports, env reset/step, actor/critic forward, and one PPO update.

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
```

Inside the allocated shell:

```bash
export DATA_ROOT=/data/$USER
export PROJECT_DIR=$DATA_ROOT/research
export CONDA_ENV=uav
export DEVICE=auto
cd "$PROJECT_DIR"

bash Lyapunov_uav/proposed/scripts/seraph/srun_ppo_update_smoke.sh
```

Expected outputs:

- `torch_cuda_available= True`
- JSON records with `event=ppo_update_smoke`
- finite PPO update stats for both `level=slow` and `level=fast`

For a short train run in the same interactive allocation:

```bash
export RUN_NAME=short_hrl_srun_smoke
export MAX_STEPS=20
export MAX_UPDATES=2
export ROLLOUT_STEPS=4
export DEVICE=auto
export REWARD_PRESET=balanced

bash Lyapunov_uav/proposed/scripts/seraph/srun_short_hrl_train.sh
```

Available reward presets are `balanced`, `conservative_queue`, and `quality_oriented`.
Use them only to prepare short smoke runs at this stage; do not compare or tune them yet.

Run sanity check after the short train:

```bash
python3 Lyapunov_uav/proposed/scripts/check_short_train_sanity.py \
  --log-path "$LOG_DIR/${RUN_NAME}.jsonl" \
  --checkpoint-dir "$CHECKPOINT_DIR"
```

Expected final line:

```text
PASS short_train_sanity
```

Create a short report for the run:

```bash
python3 Lyapunov_uav/proposed/scripts/report_short_train.py \
  --log-dir "$LOG_DIR" \
  --run-name "$RUN_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-json "$OUTPUT_DIR/${RUN_NAME}_report.json" \
  --output-md "$OUTPUT_DIR/${RUN_NAME}_report.md"
```

Diagnose reward and DPP term scales without tuning coefficients:

```bash
python3 Lyapunov_uav/proposed/scripts/analyze_reward_scale.py \
  --log-dir "$LOG_DIR" \
  --run-name "$RUN_NAME" \
  --output-json "$OUTPUT_DIR/${RUN_NAME}_scale_analysis.json" \
  --output-md "$OUTPUT_DIR/${RUN_NAME}_scale_analysis.md"
```

## 2. Submit short train with sbatch

Edit `sbatch_short_hrl_train_template.sh` before submission:

- Replace `#SBATCH --partition=TODO_PARTITION` with the Seraph GPU partition for your account.
- Confirm `CONDA_ENV` matches your environment.
- Confirm `PROJECT_DIR=/data/$USER/research` points to this repository.

The template requests one GPU, four CPU cores, 16 GB memory, and 30 minutes:

```bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
```

Submit a short train job:

```bash
export DATA_ROOT=/data/$USER
export PROJECT_DIR=$DATA_ROOT/research
export CONDA_ENV=uav
export RUN_ROOT=$DATA_ROOT/hrl_uav
export RUN_NAME=short_hrl_sbatch_smoke
export MAX_STEPS=20
export MAX_UPDATES=2
export ROLLOUT_STEPS=4
export DEVICE=auto
export REWARD_PRESET=balanced

sbatch Lyapunov_uav/proposed/scripts/seraph/sbatch_short_hrl_train_template.sh
```

Available reward presets are `balanced`, `conservative_queue`, and `quality_oriented`.

## 3. Monitor the job

```bash
squeue -u "$USER"
ls -lh "$LOG_DIR"
ls -lh "$CHECKPOINT_DIR"
tail -f "$LOG_DIR/${RUN_NAME}_console.log"
```

The Slurm stdout/stderr files are written under:

```text
/data/$USER/hrl_uav/logs/hrl-short-train-<jobid>.out
/data/$USER/hrl_uav/logs/hrl-short-train-<jobid>.err
```

After completion, run:

```bash
python3 Lyapunov_uav/proposed/scripts/check_short_train_sanity.py \
  --log-path "$LOG_DIR/${RUN_NAME}.jsonl" \
  --checkpoint-dir "$CHECKPOINT_DIR"
```

Then create a report:

```bash
python3 Lyapunov_uav/proposed/scripts/report_short_train.py \
  --log-dir "$LOG_DIR" \
  --run-name "$RUN_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-json "$OUTPUT_DIR/${RUN_NAME}_report.json" \
  --output-md "$OUTPUT_DIR/${RUN_NAME}_report.md"
```

Then diagnose reward and DPP term scales:

```bash
python3 Lyapunov_uav/proposed/scripts/analyze_reward_scale.py \
  --log-dir "$LOG_DIR" \
  --run-name "$RUN_NAME" \
  --output-json "$OUTPUT_DIR/${RUN_NAME}_scale_analysis.json" \
  --output-md "$OUTPUT_DIR/${RUN_NAME}_scale_analysis.md"
```

Check that:

- `$LOG_DIR/${RUN_NAME}.jsonl` exists.
- `$LOG_DIR/${RUN_NAME}_console.log` exists.
- `$CHECKPOINT_DIR/${RUN_NAME}_update0001.pt` exists.
- `$OUTPUT_DIR/${RUN_NAME}_report.json` exists.
- `$OUTPUT_DIR/${RUN_NAME}_report.md` exists.
- `$OUTPUT_DIR/${RUN_NAME}_scale_analysis.json` exists.
- `$OUTPUT_DIR/${RUN_NAME}_scale_analysis.md` exists.
- Sanity check ends with `PASS short_train_sanity`.

## 4. Sequential preset comparison smoke

After a single-preset short train works, run the candidate presets sequentially with
the same seed and short horizon. This is only an execution-gate comparison; do not
select final coefficients from it.

```bash
export RUN_NAME=reward_preset_compare_smoke
export MAX_STEPS=20
export MAX_UPDATES=2
export ROLLOUT_STEPS=4
export SEED=2026
export DEVICE=auto

python3 Lyapunov_uav/proposed/scripts/compare_reward_presets.py \
  --presets conservative_queue balanced quality_oriented \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --max-updates "$MAX_UPDATES" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --device "$DEVICE" \
  --run-name "$RUN_NAME" \
  --log-dir "$LOG_DIR/preset_compare" \
  --checkpoint-dir "$CHECKPOINT_DIR/preset_compare" \
  --output-dir "$OUTPUT_DIR/preset_compare"
```

Check that `$OUTPUT_DIR/preset_compare/${RUN_NAME}_summary.json` and
`$OUTPUT_DIR/preset_compare/${RUN_NAME}_summary.md` exist. Each preset also gets
its own report and scale-analysis files under `$OUTPUT_DIR/preset_compare/<preset>/`.

Then classify each candidate as `viable`, `caution`, or `reject`. This only filters
obviously unsuitable presets after short train comparison; it does not select final
coefficients.

```bash
python3 Lyapunov_uav/proposed/scripts/analyze_preset_comparison.py \
  --summary-json "$OUTPUT_DIR/preset_compare/${RUN_NAME}_summary.json" \
  --output-json "$OUTPUT_DIR/preset_compare/${RUN_NAME}_viability.json" \
  --output-md "$OUTPUT_DIR/preset_compare/${RUN_NAME}_viability.md"
```

Check that `$OUTPUT_DIR/preset_compare/${RUN_NAME}_viability.json` and
`$OUTPUT_DIR/preset_compare/${RUN_NAME}_viability.md` exist.

## 5. Multi-seed repeat short runs

After the viability analysis, repeat only `viable` and `caution` presets over a
small seed list. Presets classified as `reject` in the viability JSON are excluded
automatically. This is still a short-run stability check; do not start long
training, choose final coefficients, or compare baselines from this output.

```bash
export REPEAT_RUN_NAME=reward_preset_repeat_smoke
export REPEAT_SEEDS="0 1 2"

python3 Lyapunov_uav/proposed/scripts/repeat_reward_preset_short_runs.py \
  --viability-json "$OUTPUT_DIR/preset_compare/${RUN_NAME}_viability.json" \
  --seeds $REPEAT_SEEDS \
  --max-steps "$MAX_STEPS" \
  --max-updates "$MAX_UPDATES" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --device "$DEVICE" \
  --run-name "$REPEAT_RUN_NAME" \
  --log-dir "$LOG_DIR/preset_repeat" \
  --checkpoint-dir "$CHECKPOINT_DIR/preset_repeat" \
  --output-dir "$OUTPUT_DIR/preset_repeat"
```

Check that:

- `$OUTPUT_DIR/preset_repeat/${REPEAT_RUN_NAME}_aggregate_summary.json` exists.
- `$OUTPUT_DIR/preset_repeat/${REPEAT_RUN_NAME}_aggregate_summary.md` exists.
- Per-seed comparison and viability files exist under
  `$OUTPUT_DIR/preset_repeat/seed_<seed>/`.
- Per-preset logs/checkpoints are separated under
  `$LOG_DIR/preset_repeat/seed_<seed>/<preset>/` and
  `$CHECKPOINT_DIR/preset_repeat/seed_<seed>/<preset>/`.

## 6. Failure checklist

Import error:

- Confirm `cd "$PROJECT_DIR"` points to the repository root.
- Confirm `Lyapunov_uav/proposed` exists under `$PROJECT_DIR`.
- Run `python3 -m compileall Lyapunov_uav/proposed`.

CUDA unavailable:

- Confirm the job requested a GPU with `--gres=gpu:1`.
- Check the script output for `torch_cuda_available=`.
- If CPU-only validation is intentional, set `DEVICE=cpu`; otherwise keep `DEVICE=auto` or `DEVICE=cuda`.

Path error:

- Confirm `DATA_ROOT=/data/$USER`.
- Confirm `PROJECT_DIR=$DATA_ROOT/research`.
- Confirm `LOG_DIR`, `CHECKPOINT_DIR`, and `OUTPUT_DIR` are under `/data/$USER`.

Permission error:

- Avoid writing logs/checkpoints under `/home`.
- Run `mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR"` before submission.

Missing conda environment:

- Check `CONDA_ENV`.
- Run `conda env list`.
- If `conda` is not initialized in non-interactive Slurm shells, add the appropriate conda initialization for Seraph before `conda activate`.

Checkpoint/log directory error:

- Check the Slurm `.err` file.
- Confirm the JSONL log path and checkpoint path printed in the console log.
- Confirm there is enough quota under `/data/$USER`.

NaN/Inf sanity failure:

- Inspect the failed sanity line first.
- Check `actor_loss`, `critic_loss`, `total_reward`, `entropy`, and `dpp_terms` in the JSONL log.
- Do not tune reward coefficients from this smoke result; first reproduce with the same seed and verify env/action/observation shapes.
