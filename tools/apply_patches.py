"""Apply the ONLY differences between standardized_eval copies and their verbatim originals.

Every patch is a path-resolution shim: the copies live outside the source repo, so
__file__-relative repo discovery and a handful of hardcoded roots become env-var lookups
whose DEFAULTS are the original values. No inference or metric logic is touched.

Idempotent: re-running on already-patched files is a no-op (each patch is skipped if its
new text is already present). Run:  python3 tools/apply_patches.py
"""
import re
import sys
from pathlib import Path

SE = Path(__file__).resolve().parent.parent
VWM = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/video_world_model_new"
P3 = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/pi3_eval/evaluation/Pi3_depthpose"
G3 = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines_3D"
TAG = "# [standardized_eval PATCH]"

REPO_ENV = f'REPO = Path(os.environ.get("VWM_REPO", "{VWM}"))  {TAG} repo root via env'
METRICS_LOCAL = 'os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parent / "pi3_metrics"))'
METRICS_LOCAL_INF = 'os.environ.get("PI3_METRICS", str(Path(__file__).resolve().parents[2] / "evaluation" / "pi3_metrics"))'

PATCHES = {
 "inference/ours/run_ours_pi3.py": [
  ("REPO = Path(__file__).resolve().parent.parent\nsys.path.append(str(REPO))",
   f"{REPO_ENV}\nsys.path.append(str(REPO))"),
  ("from hydra import compose, initialize\n",
   f"from hydra import compose, initialize, initialize_config_dir  {TAG}\n"),
  ('    with initialize(version_base=None, config_path="../configurations"):',
   f'    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):  {TAG}'),
 ],
 "inference/ours/run_eval_scannetpp.py": [
  ("REPO = Path(__file__).resolve().parent.parent\nsys.path.append(str(REPO))",
   f"{REPO_ENV}\nsys.path.append(str(REPO))"),
  ("from hydra import compose, initialize\n",
   f"from hydra import compose, initialize, initialize_config_dir  {TAG}\n"),
  ('    with initialize(version_base=None, config_path="../configurations"):',
   f'    with initialize_config_dir(version_base=None, config_dir=str(REPO / "configurations")):  {TAG}'),
  (f'sys.path.insert(0, "{G3}")',
   f'sys.path.insert(0, os.environ.get("GEN3D_ROOT", "{G3}"))  {TAG}'),
  ('_PI3_DEPTH = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '              "pi3_eval/evaluation/Pi3_depthpose/utils/depth.py")',
   f'_PI3_DEPTH = {METRICS_LOCAL_INF} + "/utils/depth.py"  {TAG} verbatim local metric copy'),
 ],
 "inference/pi3/run_pi3_c50.py": [
  ('REPO = Path(__file__).resolve().parent.parent\n'
   'PI3_ROOT = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '            "pi3_eval/evaluation/Pi3_depthpose")',
   f'{REPO_ENV}\nPI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG}'),
 ],
 "inference/pi3/run_pi3_re10k50.py": [
  ('REPO = Path(__file__).resolve().parent.parent\n'
   'PI3_ROOT = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '            "pi3_eval/evaluation/Pi3_depthpose")',
   f'{REPO_ENV}\nPI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG}'),
  ('IMG_TMPL = "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50/{seq}/images"',
   'IMG_TMPL = os.environ.get("RE10K_HF50", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50")'
   f' + "/{{seq}}/images"  {TAG}'),
 ],
 "inference/da3_geo4d/run_baseline_pi3.py": [
  ("REPO = Path(__file__).resolve().parent.parent",
   REPO_ENV),
  ('PI3_DATA = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '            "pi3_eval/evaluation/Pi3_depthpose/data")',
   f'PI3_DATA = os.environ.get("PI3_ROOT", "{P3}") + "/data"  {TAG}'),
  ('GEO4D_DIR = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D"',
   f'GEO4D_DIR = os.environ.get("GEO4D_DIR", "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D")  {TAG}'),
 ],
 "inference/da3_geo4d/run_baseline_scannetpp.py": [
  ("REPO = Path(__file__).resolve().parent.parent\nsys.path.insert(0, str(REPO))",
   f"{REPO_ENV}\nsys.path.insert(0, str(REPO))"),
  ('_PI3_DEPTH = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '              "pi3_eval/evaluation/Pi3_depthpose/utils/depth.py")',
   f'_PI3_DEPTH = {METRICS_LOCAL_INF} + "/utils/depth.py"  {TAG} verbatim local metric copy'),
  ('        PI3_ROOT = ("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/"\n'
   '                    "pi3_eval/evaluation/Pi3_depthpose")',
   f'        PI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG}'),
 ],
 "evaluation/eval_ours_pi3.py": [
  (f'PI3_ROOT = "{P3}"\nsys.path.insert(0, PI3_ROOT)',
   f'PI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG} GT-data root\n'
   'sys.path.insert(0, PI3_ROOT)\n'
   f'sys.path.insert(0, {METRICS_LOCAL})  {TAG} metric imports resolve to the local verbatim copy'),
  (f'sys.path.insert(0, "{G3}")',
   f'sys.path.insert(0, os.environ.get("GEN3D_ROOT", "{G3}"))  {TAG}'),
  ("sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
   f'sys.path.insert(0, os.environ.get("VWM_REPO", "{VWM}"))  {TAG} for datasets._crop_utils'),
 ],
 "evaluation/eval_baseline_pi3.py": [
  (f'PI3_ROOT = "{P3}"\nsys.path.insert(0, PI3_ROOT)',
   f'PI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG} GT-data root\n'
   'sys.path.insert(0, PI3_ROOT)\n'
   f'sys.path.insert(0, {METRICS_LOCAL})  {TAG} metric imports resolve to the local verbatim copy'),
  ("sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
   f'sys.path.insert(0, os.environ.get("VWM_REPO", "{VWM}"))  {TAG} for datasets._crop_utils'),
 ],
 "evaluation/score_depth_hf.py": [
  ("REPO = HERE.parent.parent",
   f'REPO = Path(os.environ.get("VWM_REPO", "{VWM}"))  {TAG}'),
  (f'sys.path.insert(0, "{G3}")',
   f'sys.path.insert(0, os.environ.get("GEN3D_ROOT", "{G3}"))  {TAG}'),
  (f'PI3_ROOT = "{P3}"\nsys.path.insert(0, PI3_ROOT)',
   f'PI3_ROOT = os.environ.get("PI3_ROOT", "{P3}")  {TAG} GT-data root\n'
   'sys.path.insert(0, PI3_ROOT)\n'
   f'sys.path.insert(0, {METRICS_LOCAL})  {TAG} metric imports resolve to the local verbatim copy'),
 ],
 "evaluation/eval_re10k_dist50.py": [
  (f'PI3 = "{P3}"',
   f'PI3 = {METRICS_LOCAL}  {TAG} metric imports from local verbatim copy'),
  ('ANN = "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_pi3frames"',
   f'ANN = os.environ.get("RE10K_ANN", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_pi3frames")  {TAG}'),
  ('HF50 = "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50"',
   f'HF50 = os.environ.get("RE10K_HF50", "/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/re10k_hf50")  {TAG}'),
 ],
 "evaluation/eval_scannetpp_pose.py": [
  (f'PI3 = "{P3}"',
   f'PI3 = {METRICS_LOCAL}  {TAG} metric imports from local verbatim copy'),
 ],
}


def main():
    n_applied = n_skipped = 0
    for rel, patches in PATCHES.items():
        p = SE / rel
        src = p.read_text()
        for old, new in patches:
            if new in src:
                n_skipped += 1
                continue
            count = src.count(old)
            assert count == 1, f"{rel}: expected exactly 1 occurrence, found {count}:\n{old!r}"
            src = src.replace(old, new)
            n_applied += 1
        p.write_text(src)
        print(f"patched {rel}")
    print(f"\nDONE: {n_applied} patches applied, {n_skipped} already present.")


if __name__ == "__main__":
    main()
