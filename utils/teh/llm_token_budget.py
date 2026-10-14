"""Experiment-only native token counting; frozen Qwen code is unchanged."""
import hashlib
from pathlib import Path
from utils.teh import prompt_units

FROZEN_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

def install(model_name, model_path):
    if model_name == FROZEN_MODEL:
        return {"policy": "frozen_qwen", "model": model_name}
    from transformers import AutoTokenizer
    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if not tokenizer.chat_template and (path/"chat_template.jinja").is_file():
        tokenizer.chat_template = (path/"chat_template.jinja").read_text()
    if not tokenizer.chat_template:
        raise RuntimeError("No native chat template for "+model_name)

    class NativeCounter:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            # Frozen count helper injects Qwen's system turn. TEH actually sends
            # one user message; let the selected model's template add its defaults.
            if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
                raise RuntimeError("Unexpected frozen prompt-count message layout")
            kwargs = {"reasoning_effort":"low"} if model_name == "openai/gpt-oss-120b" else {}
            return tokenizer.apply_chat_template([messages[1]], tokenize=tokenize,
                add_generation_prompt=add_generation_prompt, **kwargs)

        def encode(self, text, add_special_tokens=False):
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    # Every existing direct/imported count function reads this same global
    # tokenizer. No function replacements, panel changes, or new truncation rules.
    prompt_units._QWEN_TOKENIZER = NativeCounter()
    prompt_units._QWEN_TOKENIZER_FAILED = None
    return {"policy":"native_chat_template_v1", "model":model_name,
            "tokenizer_path":str(path.resolve()),
            "input_cap":15360, "output_reserve":1024, "context":16384,
            "tokenizer_sha256":{name:hashlib.sha256((path/name).read_bytes()).hexdigest()
              for name in ("tokenizer.json","tokenizer.model","tokenizer_config.json","chat_template.jinja")
              if (path/name).is_file()},
            "wrapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
