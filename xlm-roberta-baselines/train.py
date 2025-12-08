from datasets import load_dataset
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

data = load_dataset("google/civil_comments", split="train")
print(data[0])

# TODO: load other datasets and concatenate them together

# make input column lowercase
def lowercase_text(row):
    row['text'] = row['text'].lower()
    return row

data_lower = data.map(lowercase_text)
print(data_lower[0])

# finetune the xlm-roberta model on this lowercase text
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# print(tokenizer.tokenize("hello world"))
model = AutoModelForMaskedLM.from_pretrained(
    "xlm-roberta-base",
    num_labels = 5,
    output_attentions = False,
    output_hidden_states = False,
)

# Optimizer & Learning Rate Scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

# Number of training epochs
epochs = 4
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)