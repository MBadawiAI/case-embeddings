import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data import load_dataset
from src.models import build_model
from src.loop import train_one_epoch, evaluate
from src.utils import set_seed

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_ds, val_ds = load_dataset(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    # Build model
    model = build_model(cfg).to(device)

    # Optimizer
    if cfg.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    # TensorBoard
    writer = SummaryWriter(cfg.train.log_dir)

    for epoch in range(cfg.train.num_epochs):
        print(f"=== Epoch {epoch} ===")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer, cfg
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, epoch, writer
        )
        print(f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
