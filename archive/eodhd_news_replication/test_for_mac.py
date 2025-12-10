# test_chronobert.py
import torch
from transformers import AutoTokenizer, AutoModel

print("Using device:", "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")  # keep CPU for now

model_name = "manelalab/chrono-bert-v1-20201231"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

inputs = tok("Test headline for ChronoBERT.", return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

print("OK, last_hidden_state:", outputs.last_hidden_state.shape)
