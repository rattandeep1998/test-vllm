from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import uvicorn
import requests

import os
app = FastAPI(title="vLLM Chat API")

# Initialize the LLM
MAX_MODEL_LEN = 384
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_lora=True, max_model_len=MAX_MODEL_LEN)

# LORA_REQUEST = LoRARequest("humor_explanation", 1, '/mnt/swordfish-pool2/horvitz/iterative_pipeline_v0/ExplanationDistillation_train_sft/42_0.0001_5_16_4_2025-03-05-17-04-54/lora_model_sft/')

class LoraRequestFields(BaseModel):
    name: str
    id: int
    path: str

class MultiChatRequest(BaseModel):
    messages: List # Batch of conversations
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = MAX_MODEL_LEN
    top_p: Optional[float] = 0.9
    model: Optional[str] = None
    lora_request: Optional[LoraRequestFields] = None

class ChatResponse(BaseModel):
    generations: List[str]

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: MultiChatRequest):
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )


        print(request.model_dump())
        
        # convert chat messages to dict
        messages = request.messages


        # print(request.messages)

        model_name = request.model

        if model_name is not None:
            print(f"Warning: model name {model_name} is not currently used")


        lora_request = None
        if request.lora_request is not None:
            lora_request = LoRARequest(
                request.lora_request.name,
                request.lora_request.id,
                request.lora_request.path,
            )

        # Process each conversation in the batch
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

        # print(outputs)

        # Extract generated text from outputs
        generations = [output.outputs[0].text for output in outputs]

        return ChatResponse(generations=generations)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# run test query on start


if __name__ == "__main__":
    try:
        response = requests.get("http://localhost:8005/health")
        if response.status_code == 200:
            print("Server is already running")
        exit(0)
    except Exception as e:
        print(e)

    uvicorn.run(app, host="0.0.0.0", port=8005)