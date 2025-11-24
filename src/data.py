import datasets
import transformers

def text_transform(text, mode):
    if mode == "none":
        return text
    if mode == "lower":
        return text.lower()
    raise ValueError(f"Unknown transform {mode}")

def load_dataset(cfg):
    ds = datasets.load_dataset(cfg.data.dataset_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.pretrained_name
    )

    def preprocess(example):
        text = text_transform(example[cfg.data.text_field],
                              cfg.data.text_transform)
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg.data.max_length,
        )
        enc["labels"] = example[cfg.data.label_field]
        return enc

    ds = ds.map(preprocess, batched=False)
    ds.set_format("torch")

    return ds["train"], ds["validation"]
