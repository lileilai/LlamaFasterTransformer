import os
from pathlib import Path

import deepspeed
import torch
import transformers
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast


CKPT_PRETRAINED = Path("/ckpt/pretrained")
model = GPTNeoXForCausalLM.from_pretrained("/data1/lileilai/gpt_neox_20b/", local_files_only=True, torch_dtype=torch.float16) #.half() (both .half() and torch_dtype=torch.float16 have this issue.
tokenizer = GPTNeoXTokenizerFast.from_pretrained(CKPT_PRETRAINED / "/data1/lileilai/gpt_neox_20b/", local_files_only=True)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
local_device = f"cuda:{local_rank}"
print(f"local device: {local_device}")

ds_engine = deepspeed.init_inference(
  model, mp_size=world_size, dtype=torch.float16, checkpoint=None,
  replace_method='auto',
  replace_with_kernel_inject=True,
)
model = ds_engine.module

# print(model.hf_device_map)
prompt = "Deepspeed is "
m_inp = tokenizer(prompt, return_tensors="pt")
attn_mask = m_inp.get("attention_mask", None).to(device=local_device)
ids = m_inp.input_ids.to(device=local_device)

with torch.no_grad():
  gen_tokens = model.generate(
    ids, attention_mask=attn_mask,
    do_sample=True, temperature=0.9, max_new_tokens=100,
    use_cache=False, # fails with use_cache=True as well
  )
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(f"generated tokens: {gen_text}")