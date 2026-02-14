# BLUFF Baseline Models

Pre-trained baseline checkpoints for BLUFF benchmark tasks.

## Available Baselines

| Model | Task 1 (F1) | Task 2 (F1) | Task 3 (F1) | Task 4 (F1) | Download |
|-------|-------------|-------------|-------------|-------------|----------|
| XLM-RoBERTa-large | -- | -- | -- | -- | [HuggingFace](https://huggingface.co/jsl5710/bluff-xlmr-large-task1) |
| mDeBERTa-v3-base | -- | -- | -- | -- | [HuggingFace](https://huggingface.co/jsl5710/bluff-mdeberta-task1) |
| Glot500 | -- | -- | -- | -- | [HuggingFace](https://huggingface.co/jsl5710/bluff-glot500-task1) |

*Results will be updated upon paper publication.*

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("jsl5710/bluff-xlmr-large-task1")
tokenizer = AutoTokenizer.from_pretrained("jsl5710/bluff-xlmr-large-task1")

inputs = tokenizer("Article text here...", return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
label = "fake" if prediction == 1 else "real"
```
