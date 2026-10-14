"""Use the evaluated frozen prompt bytes, as required for model comparison."""
import shutil
from pathlib import Path
from utils.teh import teh_runtime
from utils.teh.pics_v4 import using_pics_v4
original_setup = teh_runtime.setup_teh_run_prompts
def setup_frozen(run_dir, dataset_alias, seed_program_path, **kwargs):
    output = original_setup(run_dir, dataset_alias, seed_program_path, **kwargs)
    if using_pics_v4() and kwargs.get("pics_aamas_v0_prompt"):
        from utils.teh.pics_v4_transfer import target_run, sha
        frozen = target_run(dataset_alias) / "prompts"
        # Keep templates/runtime contracts created by the unchanged helper.
        # The evaluated registered prompt and its metadata are reused exactly.
        for name in ("infer_single_choice.txt", "prompt_meta.json"):
            shutil.copyfile(frozen/name, Path(output)/name)
        if sha(Path(output)/"infer_single_choice.txt") != sha(frozen/"infer_single_choice.txt"):
            raise RuntimeError("Frozen registered prompt copy failed")
    return output
teh_runtime.setup_teh_run_prompts = setup_frozen
