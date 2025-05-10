from multiprocessing.managers import BaseManager
import sys

# --------------------- Manager Setup ---------------------
class InferenceManager(BaseManager): pass

InferenceManager.register("QwenInference")
manager = InferenceManager(address=('localhost', 50000), authkey=b'qwenkey')
manager.connect()
qwen = manager.QwenInference()

# --------------------- Argument Parsing ---------------------
if len(sys.argv) < 2:
    print("Usage: python infer.py <image_path> [output_file] [--prompt \"your custom prompt\"]")
    sys.exit(1)

image_path = sys.argv[1]
output_path = None
custom_prompt = None

# Detect optional args
for i in range(2, len(sys.argv)):
    if sys.argv[i] == "--prompt" and i + 1 < len(sys.argv):
        custom_prompt = sys.argv[i + 1]
    elif not sys.argv[i].startswith("--"):
        output_path = sys.argv[i]

# --------------------- Prompt & Inference ---------------------
default_prompt = "Identify the chemical structure and return the SMILES inside <answer> </answer>."
prompt = custom_prompt if custom_prompt else default_prompt

result = qwen.generate(image_path, prompt)

# --------------------- Output Handling ---------------------
if output_path:
    with open(output_path, "w") as f:
        f.write(result)
else:
    print(result)
