import torch
from torch.utils.data import Dataset
import datasets
import transformers
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, cfg):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        text = text_transform(
            example[self.cfg.data.text_field],
            self.cfg.data.text_transform,
        )

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.max_length,
            return_tensors="pt",
        )

        enc = {k: v.squeeze(0) for k, v in enc.items()}

        # FIX: float → binary label
        label = example[self.cfg.data.label_field]
        enc["labels"] = torch.tensor(label >= 0.5, dtype=torch.long)
        return enc

def text_transform(text, mode):
    if mode == "none":
        return text
    if mode == "lower":
        return text.lower()
    raise ValueError(f"Unknown transform {mode}")

def load_dataset(cfg):
    ds = datasets.load_dataset(cfg.data.dataset_name)

    if cfg.data.get("max_samples"):
        ds["train"] = ds["train"].shuffle(seed=cfg.seed).select(
            range(cfg.data.max_samples)
        )
        ds["validation"] = ds["validation"].shuffle(seed=cfg.seed).select(
            range(min(cfg.data.max_samples, len(ds["validation"])))
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.pretrained_name
    )

    train_ds = TextDataset(ds["train"], tokenizer, cfg)
    val_ds = TextDataset(ds["validation"], tokenizer, cfg)

    return train_ds, val_ds


def pad_collate_fn(batch, pad_token_id = 0):    
    combined_input_ids = [sample['input_ids'] for sample in batch]
    combined_attention_masks = [sample['attention_mask'] for sample in batch]
    combined_labels = torch.stack([sample['labels'] for sample in batch])

    padded_input_ids = pad_sequence(
        combined_input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )
    padded_attention_masks = pad_sequence(
        combined_attention_masks, 
        batch_first=True, 
        padding_value=pad_token_id)
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "labels": combined_labels
    }
