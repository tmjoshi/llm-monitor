from mlx_lm import batch_generate, load

# Specify the checkpoint
checkpoint = "mlx-community/Qwen1.5-0.5B-Chat"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

# A batch of prompts
prompts = [
    "Write a story about Einstein.",
    "Why is the sky blue?",
    "What time is it?",
    "How tall is Mt Everest?",
]

# Apply the chat template and encode to tokens
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        add_generation_prompt=True,
    )
    for p in prompts
]

result = batch_generate(
    model, tokenizer, prompts, verbose=True, return_prompt_caches=True, max_tokens=2048
)

for t in result.texts:
    print(t)
    print('==================\n')