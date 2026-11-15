from mlx-lm import generate
from .qwen2 import ModelArgs

# TODO: i want the right qwen ModelArgs to be set, then the model called, metrics logged

def benchmark_inference_mlx(batch_size):
    input_len, output_len = 256, 32
    
    # TODO: set qwen2 model arguments?


    # mlx-lm generate
    response = generate(model, 
                        tokenizer,
                        prompt,
                        max_tokens,
                        verbose=True)