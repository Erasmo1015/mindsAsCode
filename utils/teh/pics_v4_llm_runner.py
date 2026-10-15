"""Run frozen teh.py with experiment-only native token counting."""
import os, runpy, sys
from pathlib import Path
sys.path.insert(0, os.environ["REPO"])
from utils.teh import pics_v4_llm_archive_paths as archive_paths
from utils.teh.llm_token_budget import install
adapter = install(os.environ["MODEL_NAME"], os.environ["MODEL_PATH"])
from utils.teh import pics_v4_llm_frozen_prompts as frozen_prompts
from utils.teh import pics_v4_dry_run
original_verify = pics_v4_dry_run.verify_report
targets = {"bergert_nosofsky_2007","1peterson2021using","guan_2020_stopping","3frey2017cct","11enkavi2019recentprobes","steyvers_2009_bandit","5speekenbrink2008learning"}
def verify_experiment(path, **kwargs):
    if kwargs.get("mode") != "official_gate" or kwargs.get("target") not in targets:
        raise RuntimeError("Certificate wrapper is scoped to approved LLM-comparison official_gate datasets")
    import json
    check = json.loads((Path(os.environ["DIFF_DIR"])/"NATIVE_FIX_CHECK.json").read_text())
    if check.get("status") != "fix_passed" or check["adapters"][os.environ["MODEL_TAG"]] != adapter:
        raise RuntimeError("Native packing adapter differs from focused CPU check")
    return original_verify(os.environ["CERTIFICATE"], **kwargs)
pics_v4_dry_run.verify_report = verify_experiment
runpy.run_path(str(Path(os.environ["REPO"])/"teh.py"), run_name="__main__")
