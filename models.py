import os
import json
from re import M
import pprint
import argparse
import sys
from packaging import version
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pprint
import torch
from collections import OrderedDict
from load_data import CLASSES_DICT, CLASSES_LIST, THRESHOLDS, make_dataloader_with_thresholds
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel


import torch
import torch.nn as nn
import torch.nn.functional as F

def build_head_by_layers(hidden_size: int, num_labels: int, layers: int, dropout_p: float = 0.1) -> nn.Module:
    # Map "layers" -> list of hidden layer sizes (excluding input and output)
    hidden_layer_sizes_map = {
        1: [],
        2: [512],
        3: [1024, 512],
        4: [1024, 512, 256],
        5: [1024, 512, 256, 128],
    }

    try:
        hidden_layer_sizes = hidden_layer_sizes_map[layers]
    except KeyError:
        raise ValueError("We only support 1, 2, 3, 4, or 5 layers.")

    # Build full sequence of dimensions: input -> hidden... -> output
    dims = [hidden_size] + hidden_layer_sizes + [num_labels]

    modules = OrderedDict()
    linear_idx = 1

    for i in range(len(dims) - 1):
        in_dim, out_dim = dims[i], dims[i + 1]
        modules[f'linear{linear_idx}'] = nn.Linear(in_dim, out_dim)

        # Add activation + dropout after every linear except the last
        if i < len(dims) - 2:
            modules[f'relu{linear_idx}'] = nn.ReLU()
            if dropout_p > 0:
                modules[f'dropout{linear_idx}'] = nn.Dropout(p=dropout_p)

        linear_idx += 1

    return nn.Sequential(modules)

# has to inherit 
class ClassificationModel(nn.Module):
    def __init__(self, backbone, classes, head_layers):
        super().__init__()

        self.backbone = backbone
        self.classes = classes
        self.head_layers = head_layers

        # Infer hidden size from backbone config if available, or require explicit attr
        if hasattr(backbone, "config") and hasattr(backbone.config, "hidden_size"):
            hidden_size = backbone.config.hidden_size
        elif hasattr(backbone, "hidden_size"):
            hidden_size = backbone.hidden_size
        else:
            raise ValueError(
                "Could not infer hidden_size from backbone. "
                "Add a `hidden_size` attribute or customize this code."
            )
        
        heads = {}
        for head_name, num_labels in classes.items():
            layers = head_layers.get(head_name, 1)  # default to 1 layer if not specified
            heads[head_name] = build_head_by_layers(
                hidden_size=hidden_size,
                num_labels=num_labels,
                layers=layers,
            )
        self.heads = nn.ModuleDict(heads)
    
    def forward(self, **kwargs):
        """
        Assumes HuggingFace-style backbone:
            outputs = backbone(**kwargs)
            h = outputs.last_hidden_state[:, 0, :]
        Returns dict[head_name] -> logits (B, num_labels)
        """
        outputs = self.backbone(**kwargs)                 
        h = outputs.last_hidden_state[:, 0, :]            

        out = {}
        for head_name, head in self.heads.items():
            out[head_name] = head(h)                     
        return out
    

    def save(self, save_dir: str):
        """
        Saves:
          - backbone weights under save_dir/backbone/
          - heads weights as save_dir/heads.pt
          - config (classes + head_layers) as save_dir/heads_config.json
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1) Save backbone
        backbone_dir = os.path.join(save_dir, "backbone")
        os.makedirs(backbone_dir, exist_ok=True)

        if hasattr(self.backbone, "save_pretrained"):
            # HF-style model
            self.backbone.save_pretrained(backbone_dir)
        else:
            torch.save(
                self.backbone.state_dict(),
                os.path.join(backbone_dir, "backbone.pt"),
            )

        # 2) Save heads (state_dict per head)
        heads_path = os.path.join(save_dir, "heads.pt")
        torch.save({name: head.state_dict() for name, head in self.heads.items()},
                   heads_path)

        # 3) Save heads config
        config = {
            "classes": self.classes,        
            "head_layers": self.head_layers, 
        }
        config_path = os.path.join(save_dir, "heads_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_backbone(self, checkpoint: str):
        """
        Load backbone weights.

        `checkpoint` can be:
          - A directory created by `ClassificationModel.save(save_dir)` (contains `backbone/`)
          - A HuggingFace model name or local HF directory
          - A plain PyTorch state_dict file (e.g. 'backbone.pt')
        """
        # Case 1: our own saved layout: <checkpoint>/backbone/...
        backbone_dir = os.path.join(checkpoint, "backbone")
        if os.path.isdir(checkpoint) and os.path.isdir(backbone_dir):
            # Prefer HF-style if possible
            if hasattr(self.backbone.__class__, "from_pretrained"):
                self.backbone = self.backbone.__class__.from_pretrained(backbone_dir)
            else:
                ckpt_path = os.path.join(backbone_dir, "backbone.pt")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                self.backbone.load_state_dict(state_dict)
            return

        # Case 2: try HuggingFace API directly (handles local or hub)
        if hasattr(self.backbone.__class__, "from_pretrained"):
            # This will work if `checkpoint` is either a local HF dir or a hub ID
            self.backbone = self.backbone.__class__.from_pretrained(checkpoint)
            return

        # Case 3: fallback to plain torch.load on whatever path we got
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.backbone.load_state_dict(state_dict)
        
    def load_checkpoint(self, checkpoint_dir: str):
        # loads both the backbone and the heads
        """
        Loads both backbone and heads from a directory created by `.save`.

        If you point this at a pure HF model name/path (no heads.pt), it will:
          - Load backbone from HF
          - Leave heads randomly initialized (since there is no heads.pt)
        """
        # 1) Load backbone (this already handles HF/local logic)
        self.load_backbone(checkpoint_dir)

        # 2) Try to load heads if they exist (our custom layout)
        heads_path = os.path.join(checkpoint_dir, "heads.pt")
        if not os.path.isfile(heads_path):
            # No heads saved here; that's fine (e.g., HF-only checkpoint)
            return

        heads_state = torch.load(heads_path, map_location="cpu")
        for name, state_dict in heads_state.items():
            if name not in self.heads:
                raise KeyError(
                    f"Head '{name}' in checkpoint but not in current model. "
                    f"Current heads: {list(self.heads.keys())}"
                )
            self.heads[name].load_state_dict(state_dict)



def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str = "bce",
    pos_weight=None,
    focal_gamma: float = 2.0,
) -> float:
    """
    Train one epoch with a configurable loss function.

    Args:
        model:       ClassificationModel (or similar) returning dict[head_name -> (B,1) logits].
        dataloader:  yields batches with keys: "input_ids", "attention_mask", and one
                     float target per head in CLASSES_DICT (e.g. "toxicity").
        optimizer:   torch optimizer.
        device:      torch.device.
        loss_type:   one of {"bce", "bce_pos_weight", "focal", "focal_pos_weight"}.
        pos_weight:  None, scalar, or dict[head_name -> float] used for *_pos_weight modes.
        focal_gamma: gamma parameter for focal loss.

    Returns:
        Average loss over the epoch.
    """

    def get_pos_weight_for_head(head_name: str, batch_size: int, device: torch.device):
        """Return a pos_weight tensor of shape (1,) on the right device."""
        if pos_weight is None:
            w = 1.0
        elif isinstance(pos_weight, dict):
            w = pos_weight.get(head_name, 1.0)
        else:
            # scalar shared for all heads
            w = float(pos_weight)
        return torch.tensor([w], device=device)

    def focal_loss_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float,
        pos_w: float | None = None,
    ) -> torch.Tensor:
        """
        Binary focal loss with logits.
        logits:  (B,1)
        targets: (B,1) in {0,1} or [0,1]
        pos_w:   optional scalar >0, weight for positive examples.
        """
        # BCE per example
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B,1)

        # p = sigmoid(logits); p_t = p if y=1 else 1-p
        p = torch.sigmoid(logits)
        p_t = torch.where(targets >= 0.5, p, 1.0 - p)  # (B,1)

        focal_factor = (1.0 - p_t).pow(gamma)  # (B,1)
        loss = focal_factor * bce  # (B,1)

        if pos_w is not None and pos_w != 1.0:
            # upweight positives
            weight = torch.ones_like(loss)
            weight = torch.where(targets >= 0.5, pos_w * weight, weight)
            loss = weight * loss

        return loss.mean()

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits_dict = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = 0.0

        for head_name in CLASSES_DICT.keys():
            # targets: (B,) -> (B,1)
            targets = batch[head_name].to(device).float().unsqueeze(-1)  # (B,1)
            head_logits = logits_dict[head_name]                         # (B,1)

            if loss_type == "bce":
                loss += F.binary_cross_entropy_with_logits(
                    head_logits, targets, reduction="mean"
                )

            elif loss_type == "bce_pos_weight":
                pw = get_pos_weight_for_head(head_name, head_logits.size(0), device)
                loss += F.binary_cross_entropy_with_logits(
                    head_logits, targets, pos_weight=pw, reduction="mean"
                )

            elif loss_type == "focal":
                loss += focal_loss_with_logits(
                    head_logits, targets, gamma=focal_gamma, pos_w=None
                )

            elif loss_type == "focal_pos_weight":
                pw = get_pos_weight_for_head(head_name, head_logits.size(0), device)
                loss += focal_loss_with_logits(
                    head_logits, targets, gamma=focal_gamma, pos_w=pw.item()
                )

            else:
                raise ValueError(
                    f"Unknown loss_type '{loss_type}'. "
                    "Expected one of: 'bce', 'bce_pos_weight', 'focal', 'focal_pos_weight'."
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":
    # ---- 1. Setup backbone + tokenizer from HF ----
    model_name = "FacebookAI/xlm-roberta-base"  # or "FacebookAI/xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    backbone = XLMRobertaModel.from_pretrained(model_name)

    # ---- 2. Define heads: head_name -> num_labels / layers ----
    classes = CLASSES_DICT
    print(classes)
    head_layers = {
        k: 2 for k in classes
    }

    model = ClassificationModel(backbone, classes, head_layers)
    model.eval()  # disable dropout for deterministic testing

    # ---- 3. Dummy batch through the model ----
    texts = [
        "This is a harmless example sentence.",
        "You are the worst driver I've ever seen.",
        "She's dumb aas a brick.",
    ]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**enc)

    print("=== Forward pass shapes ===")
    for head_name, head_logits in logits.items():
        print(f"{head_name}: {head_logits.shape}")
        print(f"{head_logits}")

    # ---- 4. Save to disk ----
    save_dir = "./tmp_xlm_roberta_model_with_heads"
    print(f"\nSaving model to {save_dir} ...")
    model.save(save_dir)

    # ---- 5. Reload from our checkpoint dir (backbone + heads) ----
    backbone2 = XLMRobertaModel.from_pretrained(model_name)
    model2 = ClassificationModel(backbone2, classes, head_layers)
    model2.eval()
    model2.load_checkpoint(save_dir)

    with torch.no_grad():
        logits2 = model2(**enc)

    print("\n=== Reloaded checkpoint check ===")
    for head_name in logits.keys():
        same = torch.allclose(logits[head_name], logits2[head_name], atol=1e-6)
        print(f"{head_name}: outputs match after load_checkpoint? {same}")

    # ---- 6. Optional: test HF-style loading path directly ----
    # Here we ignore our saved backbone and just load from HF name/path
    backbone3 = XLMRobertaModel.from_pretrained(model_name)
    model3 = ClassificationModel(backbone3, classes, head_layers)
    model3.eval()
    model3.load_backbone(model_name)  # this calls from_pretrained(model_name) internally

    with torch.no_grad():
        logits3 = model3(**enc)

    print("\n=== HF-name loading sanity check (shapes only) ===")
    for head_name, head_logits in logits3.items():
        print(f"{head_name}: {head_logits.shape}")
    ds = load_dataset("google/civil_comments")
    print(ds['train'][0])
    train_loader = make_dataloader_with_thresholds(
        ds["train"],
        thresholds=THRESHOLDS,
        batch_size=32,
        shuffle=True,
    )
    
    test_loader = make_dataloader_with_thresholds(
        ds["test"],
        thresholds=THRESHOLDS,
        batch_size=32,
        shuffle=True,
    )

    validation_loader = make_dataloader_with_thresholds(
        ds["validation"],
        thresholds=THRESHOLDS,
        batch_size=32,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)



    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device='cuda',
        loss_type="bce",
)
