from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from torch.utils.data import DataLoader


# ds has three splits: train, test, validation of length 1804874, 97320. 97320

# each input has the keys 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexually_explicit'

# CLASSES_LIST = [
#     'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexually_explicit'
# ]
CLASSES_DICT = {
    'toxicity' : 1, 
    'severe_toxicity' : 1, 
    'threat' : 1, 
    'obscene': 1, 
    'insult': 1, 
    'identity_attack': 1, 
    'sexual_explicit': 1
}

THRESHOLDS = {
    "toxicity": 0.5,
    "severe_toxicity": 0.5,
    "obscene": 0.5,
    "threat": 0.5,
    "insult": 0.5,
    "identity_attack": 0.5,
    "sexual_explicit": 0.5,
}


CLASSES_LIST = list(CLASSES_DICT.keys())

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

ds = load_dataset("google/civil_comments")
def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    # copy float labels through unchanged
    for head_name in CLASSES_DICT.keys():
        enc[head_name] = batch[head_name]
    return enc

tokenized = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds["train"].column_names,  # removes 'text' and original fields
)

# now tell HF: these columns should be returned as torch.Tensors
columns = ["input_ids", "attention_mask"] + list(CLASSES_DICT.keys())
tokenized.set_format(type="torch", columns=columns)

from torch.utils.data import DataLoader
import torch

THRESHOLDS = {
    "toxicity": 0.5,
    "severe_toxicity": 0.5,
    "obscene": 0.5,
    "threat": 0.5,
    "insult": 0.5,
    "identity_attack": 0.5,
    "sexual_explicit": 0.5,
}

def make_dataloader_with_thresholds(dataset_split, thresholds, batch_size=32, shuffle=False):
    def collate_fn(examples):
        batch = {}

        # examples[0] is now all tensors: safe to stack everything
        for key in examples[0].keys():
            batch[key] = torch.stack([ex[key] for ex in examples])

        # add *_bin labels based on thresholds
        for head_name in CLASSES_DICT.keys():
            scores = batch[head_name].float()              # (B,)
            thr = thresholds.get(head_name, 0.5)
            batch[head_name + "_bin"] = (scores >= thr).long()  # (B,)

        return batch

    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

if __name__ == "__main__":
    print(len(ds['train']))
    print(len(ds['test']))
    print(len(ds['validation']))
    print(ds['train'][456456])