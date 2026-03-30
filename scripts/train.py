import hydra
from omegaconf import DictConfig
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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
    use_cuda = torch.cuda.is_available()

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    train_ds, val_ds = load_dataset(cfg)
    rank, world_size, device = setup()

    # Setup DistributedSampler for training dataset
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,  # shuffle is handled by DistributedSampler
        num_workers=cfg.train.dataloader.num_workers,
        pin_memory=cfg.train.dataloader.pin_memory and use_cuda,
        persistent_workers=cfg.train.dataloader.persistent_workers,
        drop_last=cfg.train.dataloader.drop_last,
        prefetch_factor=(
            cfg.train.dataloader.prefetch_factor
            if cfg.train.dataloader.num_workers > 0
            else None
        ),
        sampler=train_sampler,  # Use the sampler
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.dataloader.num_workers,
        pin_memory=cfg.train.dataloader.pin_memory and use_cuda,
    )

    # ------------------------------------------------------------
    # Build model using DDP
    # ------------------------------------------------------------
    # rank, world_size, device = setup()
    model = build_model(cfg).to(device)
    model = DDP(model, device_ids=[device.index], output_device=device)

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
        train_sampler.set_epoch(epoch)  # Update sampler at the beginning of each epoch

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

def setup():
    dist.init_process_group(backend=os.environ.get("BACKEND", "gloo"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, device

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
