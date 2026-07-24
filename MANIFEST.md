# MANIFEST — source → copy map

Copied 2026-07-18. Most copies are verbatim except the path shims in
tools/apply_patches.py and the optional OURS component integration noted below.
MD5s are of the source files at copy time.

| copy | source | source md5 |
|---|---|---|
| inference/ours/run_ours_pi3.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_ours_pi3.py | 7160c6db398f791020148616e5d7c775 |
| inference/ours/run_eval_scannetpp.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_eval_scannetpp.py | 8643cac893bca654de05cfd23f70a056 |
| inference/pi3/run_pi3_c50.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_pi3_c50.py | 8771fd8854e2b06c4543c315f001b49f |
| inference/pi3/run_pi3_re10k50.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_pi3_re10k50.py | 73a4527a91232da194ce131d2b0beae3 |
| inference/da3_geo4d/run_baseline_pi3.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_baseline_pi3.py | a459015a26b3bc22fe099665eef6fee0 |
| inference/da3_geo4d/run_baseline_scannetpp.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/run_baseline_scannetpp.py | e51d43eaeccd7577d61caf26a4fb6a1b |
| evaluation/eval_ours_pi3.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/eval_ours_pi3.py | 8e502e92cdcf2aa0b575dd4200617012 |
| evaluation/eval_baseline_pi3.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/eval_baseline_pi3.py | de1a6615702fcdb257936645774f68d4 |
| evaluation/score_depth_hf.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/hf/score_depth_hf.py | a39957f1d61cf700d02412c5ca69decc |
| evaluation/align_ablation.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/hf/align_ablation.py | c879c305bf06712ea62bb40516be60c1 |
| evaluation/eval_re10k_dist50.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/hf/eval_re10k_dist50.py | 9c93998c557c1905865d5e12b5e5b603 |
| evaluation/eval_scannetpp_pose.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new/eval_pi3/hf/eval_scannetpp_pose.py | ced1b1097bc716d3a3d65142cac69dbd |
| evaluation/pi3_metrics/relpose/evo_utils.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose/relpose/evo_utils.py | fae5cd50f553c55548f1c2b82db77895 |
| evaluation/pi3_metrics/utils/depth.py | /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose/utils/depth.py | 36ac5b77a5d6fccc6dbb8d277fdb91eb |

Notes:
- evaluation/pi3_metrics files are UNPATCHED (md5-identical to source): the π³ metric code is untouched.
- `inference/ours/run_ours_pi3.py` and `run_eval_scannetpp.py` now extend the
  copied bridges with opt-in depth-VAE-v2, shared-intrinsics, confidence-output,
  and verified-BA support. Their no-flag path retains the copied behavior.
- `inference/ours/benchmark_components.py` is new integration code, not a
  campaign copy. It uses the parent checkout's tracked
  `scripts/shared_intrinsics_camera.py` and checksum-pinned CPython 3.10 BA
  core (`07d4c86a…4606c`).
- `evaluation/eval_ours_pi3.py` only changes its scorer scratch filename to be
  process-specific for concurrent array jobs; scoring logic is unchanged.
- align_ablation.py is score_depth_hf's sibling dependency (numpy-only, unpatched).
