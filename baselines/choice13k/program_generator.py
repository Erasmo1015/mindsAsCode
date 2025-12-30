import re
from typing import List


class Choice13kProgramGenerator:
    def __init__(self, client, model_name: str, max_tokens: int = 800):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_programs(self, prompt: str, n_programs: int) -> List[str]:
        programs = []
        for _ in range(n_programs):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
                match = re.search(r"```python(.*?)```", content, re.DOTALL | re.IGNORECASE)
                code = match.group(1).strip() if match else content.strip()
                programs.append(code)
            except Exception:
                programs.append("")
        return programs

