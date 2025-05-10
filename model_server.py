from multiprocessing.managers import BaseManager
import torch
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

CHECKPOINT = 25000
QWEN_MODEL = "2"
PARAM = "2B"
MODEL_NAME = f"Qwen/Qwen{QWEN_MODEL}-VL-{PARAM}-Instruct"
CHECKPOINT_PATH = f"Qwen{QWEN_MODEL}-VL-{PARAM}-SFT-Instruct-Full/checkpoint-{CHECKPOINT}"
OUTPUT_JSON = f"Qwen{QWEN_MODEL}-{PARAM}_sft_eval_results_{CHECKPOINT}_Full.json"

class QwenInferenceWorker:
    def __init__(self, model_name, checkpoint_path):
        print("Loading model...")
        self.processor = Qwen2VLProcessor.from_pretrained(model_name, use_fast=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16).to("cuda")
        self.model.eval()
        print("Model ready.")

        self.system_prompt = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    def generate(self, image_path, prompt):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        if not image_inputs:
            return "[ERROR] Invalid image"
        inputs = self.processor(text=[prompt_text], images=[image_inputs[0]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            prediction = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return prediction

class InferenceManager(BaseManager): pass

if __name__ == "__main__":
    worker = QwenInferenceWorker(MODEL_NAME, CHECKPOINT_PATH)
    InferenceManager.register("QwenInference", callable=lambda: worker)
    manager = InferenceManager(address=('', 50000), authkey=b'qwenkey')
    print("🚀 Model server is running...")
    manager.get_server().serve_forever()
