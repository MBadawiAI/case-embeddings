import torch
from torch.utils.data import Dataset
import datasets
import transformers

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

