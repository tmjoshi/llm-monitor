from mlx-lm import generate
from .qwen2 import ModelArgs

# TODO: i want the right qwen ModelArgs to be set, then the model called, metrics logged

def benchmark_inference_mlx(batch_size):
    model = '/Users/tejas/projects/llm-monitor/qwen2.py' # generate does not take filepath, but model object
    prompt = 'What makes the sun red and warm?'
    input_len, output_len = 256, 32
    
    # TODO: set qwen2 model arguments?
    ModelArgs.model_type = "qwen2"
    ModelArgs.hidden_size = 1024
    ModelArgs.num_hidden_layers = 24
    ModelArgs.intermediate_size = 2816
    ModelArgs.num_attention_heads = 16
    ModelArgs.rms_norm_eps = 1e-06
    ModelArgs.vocab_size = 151936
    ModelArgs.num_key_value_heads = 16
    ModelArgs.tie_word_embeddings = False


    # mlx-lm generate
    response = generate(model, 
                        tokenizer,
                        prompt,
                        max_tokens=output_len,
                        verbose=True)