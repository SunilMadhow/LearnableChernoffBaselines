import torch
from transformers import AutoTokenizer, AutoModel

from diffusion_core import ModelConfig
from llada_core import LLaDACore

# --------------------------------------------------
# 1. Device: CUDA
# --------------------------------------------------
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
print("Using device:", device)

# --------------------------------------------------
# 2. Load tokenizer + model
# --------------------------------------------------
hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    hf_model_id,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    hf_model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model.to(device)
model.eval()

print("Model device:", next(model.parameters()).device)

# --------------------------------------------------
# 3. Prompt
# --------------------------------------------------
prompt_text = "Write a poem about the ocean without using the letter e"
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
prompt_len = len(prompt_ids)

print("Prompt length:", prompt_len)

# --------------------------------------------------
# 4. Diffusion configuration


S = 1                        # single sequence so logs are readable
N = 1
print("changed temperature")

T_steps = 16                 
gen_len = 16                 
max_length = prompt_len + gen_len
cfg = ModelConfig()
cfg.model_name = "llada"
cfg.state_shape = (max_length,)
cfg.index_set = list(range(T_steps + 1))
cfg.deets = {
    "steps": T_steps,
    "hf_model_id": hf_model_id,
    "mask_id": 126336,
    "temperature": 0.8,      # Recommendation: Set to 0 for logic/math, 0.5-0.8 for creative
    "cfg_scale": 0.0,
    "remasking": "low_confidence",
    "prompt_len": prompt_len,
    "block_len": 4,         # <--- ADD THIS LINE (Must be < gen_len)
}

# --------------------------------------------------
# 5. Instantiate core
# --------------------------------------------------
core = LLaDACore(
    cfg,
    tokenizer,
    model,
    S=S,
    N=N,
)

core.set_prompt(prompt_ids)

# --------------------------------------------------
# 6. Manual diffusion loop with printing
# --------------------------------------------------
core.reset()

mask_id = cfg.deets["mask_id"]

print("\n================ DIFFUSION TRAJECTORY ================\n")

while core.t > 0:
    t = core.t

    # propose next states
    proposals = core.propose()
    next_state = proposals[:, 0, :]   # unguided: take first proposal

    # commit
    core.commit(next_state)

    # inspect state
    tokens = core.X[0]
    num_masks = int((tokens == mask_id).sum())

    decoded = tokenizer.decode(
        tokens,
        skip_special_tokens=False,
    )

    print(f"\n--- t = {t:02d} | remaining masks = {num_masks} ---\n")
    print(decoded)

print("\n================ END DIFFUSION =================\n")
