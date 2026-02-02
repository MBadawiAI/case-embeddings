import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from src.data import load_dataset
from src.models import build_model
from src.loops import train_one_epoch, evaluate
from src.utils import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    train_ds, val_ds = load_dataset(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.dataloader.shuffle,
        num_workers=cfg.train.dataloader.num_workers,
        pin_memory=cfg.train.dataloader.pin_memory and use_cuda,
        persistent_workers=cfg.train.dataloader.persistent_workers,
        drop_last=cfg.train.dataloader.drop_last,
        prefetch_factor=(
            cfg.train.dataloader.prefetch_factor
            if cfg.train.dataloader.num_workers > 0
            else None
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.dataloader.num_workers,
        pin_memory=cfg.train.dataloader.pin_memory and use_cuda,
    )

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    model = build_model(cfg).to(device)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    if cfg.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer type {cfg.optimizer.type}")

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.num_epochs
    warmup_steps = int(cfg.scheduler.warmup_ratio * total_steps)

    if cfg.scheduler.type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=cfg.scheduler.num_cycles,
        )
    else:
        raise ValueError(f"Unknown scheduler type {cfg.scheduler.type}")

    # ------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------
    writer = SummaryWriter(cfg.train.log_dir)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(cfg.train.num_epochs):
        print(f"=== Epoch {epoch + 1}/{cfg.train.num_epochs} ===")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            writer=writer,
            cfg=cfg,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer,
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
