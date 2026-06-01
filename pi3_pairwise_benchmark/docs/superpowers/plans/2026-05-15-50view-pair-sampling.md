# 50-View Pair Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the Pi3 benchmark harness from fixed 10-frame smoke samples to 50-frame sequence evaluation and deterministic sampled 2-view evaluation.

**Architecture:** Keep Pi3's official multi-view evaluator as the source of truth for N-view reporting. Add local manifest and pair-sampling utilities so 2-view Pi3/VWM/Geo4D adapters use the same sampled pair lists and metrics.

**Tech Stack:** Python, NumPy, unittest, Pi3 official `relpose/eval_angle.py`, Slurm wrappers already present in `pi3_pairwise_benchmark`.

---

### Task 1: Generalize Manifests

**Files:**
- Modify: `pi3_pairwise_benchmark/io.py`
- Modify: `tests/test_io.py`

- [ ] Update `ManifestRow.from_json` to accept any `N >= 2` instead of exactly 10 images.
- [ ] Add tests for 50-frame manifests and mismatched image/pose counts.
- [ ] Run `python -m unittest pi3_pairwise_benchmark/tests/test_io.py -v`.

### Task 2: Add Pair Sampling Policy

**Files:**
- Create: `pi3_pairwise_benchmark/sampling.py`
- Create: `tests/test_sampling.py`

- [ ] Implement `select_pairs` with policies `all`, `short`, `medium`, `long`, and `mixed`.
- [ ] Use unordered pairs only, matching Pi3 `torch.combinations`.
- [ ] Make random sampling deterministic from `seed` and sequence name.
- [ ] Run `python -m unittest pi3_pairwise_benchmark/tests/test_sampling.py -v`.

### Task 3: Wire Sampling Into 2-View Evaluators

**Files:**
- Modify: `scripts/run_pi3_2view_pairs.py`
- Modify: `scripts/run_ours_2view_pairs.py`

- [ ] Replace `pairwise_indices(10)` with `select_pairs(len(row.image_paths), ...)`.
- [ ] Add CLI flags for pair policy, pairs per sequence, seed, and gap ranges.
- [ ] Preserve default behavior as all pairs.
- [ ] Run py_compile on both scripts.

### Task 4: Add 50-View Seq-Map Builder

**Files:**
- Create: `scripts/make_re10k_contiguous_seq_map.py`
- Create: `tests/test_make_re10k_contiguous_seq_map.py`

- [ ] Read prepared/source Re10K sequence `annotations.json`.
- [ ] Select 50 contiguous or evenly spaced continuous-window frame IDs per valid sequence.
- [ ] Emit a Pi3-compatible seq-id-map JSON and seq file.
- [ ] Run the new unit test.

### Task 5: Add 50-View Slurm/Wrapper Entry

**Files:**
- Create: `scripts/run_re10k_pose_50view_smoke.sh`
- Create: `slurm/re10k_pose_50view_smoke_gpu.sbatch`
- Modify: `README.md`
- Modify: `outputs/job_status.md`

- [ ] Prepare 50-frame Re10K smoke data.
- [ ] Run Pi3 official evaluator once per 50-frame sequence.
- [ ] Materialize a 50-frame manifest for VWM/Geo4D adapters.
- [ ] Document that 50-view reporting uses all `C(50,2)=1225` unordered pairs after one inference.

### Task 6: Comparison Visualization

**Files:**
- Create: `scripts/build_pairwise_comparison_visualizations.py`

- [ ] Load manifest plus Pi3 and ours prediction directories.
- [ ] Render input pair, GT+Pi3 pose, GT+ours pose, metrics, and PLY frustums.
- [ ] Include optional depth/point-map thumbnails if files exist.
- [ ] Document output path in README.
