from vllm import LLM, SamplingParams
import os
import torch

print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["LOCAL_RANK"] = "0"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")