from mlx_lm import generate, load

path = 
model, tokenizer = load(path_or_hf_repo=path)

prompt = "Why is the sky blue?"
conversation = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    conversation=conversation,
    add_generation_prompt=True,
)

max_tokens = 1_000
verbose = True

response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    verbose=verbose,
)