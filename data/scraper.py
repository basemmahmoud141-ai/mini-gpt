import json
import re
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

print(f"Scanning folder for text files: {project_root}")

txt_files = list(project_root.glob("*.txt")) + list(script_dir.glob("*.txt"))

if not txt_files:
    raise FileNotFoundError(
        f"Could not find ANY text (.txt) files in '{project_root}'. "
        "Please make sure your text file is pasted or saved inside the MINI_GPT folder."
    )

input_file = txt_files[0]
for f in txt_files:
    if f.name.lower() == "data.txt":
        input_file = f
        break

output_file = project_root / "data.jsonl"

print(f"Found text file: {input_file.resolve()}")
print("Converting text to JSONL format (one story per line)...")

# Phrases that commonly signal the START of a new story in this corpus.
# NOTE: this is a heuristic, not a guarantee — the source file has no
# explicit story delimiter, so boundaries are inferred from wording.
STORY_OPENERS = re.compile(
    r"^(Once upon a time|One day|One sunny day|One morning|One afternoon|"
    r"One evening|One night|One time|There was once|There once was|"
    r"It was a (sunny|rainy|cold|warm|beautiful) day)",
    re.IGNORECASE,
)

with open(input_file, "r", encoding="utf-8") as fin:
    content = fin.read()

paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

stories = []
current = []

for para in paragraphs:
    if STORY_OPENERS.match(para) and current:
        # Looks like a new story is starting — flush the current one.
        stories.append(" ".join(current))
        current = [para]
    else:
        current.append(para)

if current:
    stories.append(" ".join(current))

count = 0
with open(output_file, "w", encoding="utf-8") as fout:
    for story in stories:
        record = {"text": story}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"Done! Successfully converted {count} stories into '{output_file.resolve()}'")
print("NOTE: story boundaries were inferred heuristically and may not be 100% accurate.")