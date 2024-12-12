import vllm

def initialize_model(model_path, quantization="awq", tensor_parallel_size=2, gpu_memory_utilization=0.95):
    return vllm.LLM(
        model_path=model_path,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=5000,
        disable_log_stats=False,
        enable_prefix_caching=True
    )

def generate_responses(model, input_texts, sampling_params):
    responses = model.generate(input_texts, sampling_params, use_tqdm=True)
    return [x.outputs[0].text for x in responses]

import requests

# Function to send requests to the vLLM server
def generate_responses(input_texts, model_name="Qwen/Qwen2.5-7B-Instruct", server_url="http://localhost:8000/v1/chat/completions"):
    responses = []
    for text in input_texts:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": text}]
        }
        response = requests.post(server_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            responses.append(data['choices'][0]['message']['content'])
        else:
            responses.append(f"Error: {response.status_code}, {response.text}")
    return responses
