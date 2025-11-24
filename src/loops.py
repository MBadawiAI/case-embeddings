import torch

def train_one_epoch(model, loader, optimizer, device, epoch, writer, cfg):
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg.train.max_grad_norm)
        optimizer.step()

        # Logging
        writer.add_scalar("train/loss", loss.item(),
                          epoch * len(loader) + step)

        # accuracy
        preds = out.logits.argmax(dim=-1)
        labels = batch["labels"]
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        preds = out.logits.argmax(dim=-1)
        labels = batch["labels"]

        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", acc, epoch)

    return avg_loss, acc
