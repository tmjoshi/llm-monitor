from mlx-lm import generate
from llm-monitor.qwen2 import ModelArgs

def benchmark_inference_mlx(batch_size):
    input_len, output_len = 256, 32
    
    # set qwen2 model arguments

    # mlx-lm generate
    response = generate(model, 
                        tokenizer,
                        prompt,
                        max_tokens,
                        verbose)