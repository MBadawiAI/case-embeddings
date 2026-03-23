import transformers

def build_model(cfg):
    if cfg.model.task_type == "classification":
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            cfg.model.pretrained_name,
            num_labels=cfg.model.num_labels,
            device_map="cpu",
        )
    raise ValueError(f"Unknown task_type {cfg.model.task_type}")
