from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

model_name = "ProsusAI/finbert"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = ["Stocks rallied on strong earnings.", "Company warns of lower profits."]
inputs = tok(text, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = F.softmax(logits, dim=-1)
print("Probabilities:", probs)
