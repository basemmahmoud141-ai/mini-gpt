import os
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

from model.gpt import GPTModel
from tokenizer import create_jsonl_dataloaders
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def calc_loss_batch(input_batch, target_batch, model, device, autocast_device_type, dtype):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)

    with torch.autocast(device_type=autocast_device_type, dtype=dtype):
        logits = model(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


@torch.no_grad()
def evaluate_loss(model, dataloader, device, autocast_device_type, dtype):
    model.eval()
    total_loss = 0.0
    if len(dataloader) == 0:
        return 0.0
    pbar = tqdm(dataloader, desc="Evaluating Validation Loss", leave=False)
    for input_batch, target_batch in pbar:
        loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_device_type, dtype)
        total_loss += loss.item()
        
    model.train()
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, loss, history, filename="gpt_checkpoint.pt"):
    raw_model = model.module if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": loss,
        "history": history,
    }
    torch.save(checkpoint, filename)


def plot_loss_curve(epochs, train_losses, val_losses, tokens_seen):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_losses, label="Training loss", color="tab:blue", linestyle="-", linewidth=2, marker="o")
    ax1.plot(epochs, val_losses, label="Validation loss", color="tab:orange", linestyle="-.", linewidth=2, marker="o")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=12)
    ax1.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close(fig)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        autocast_device_type = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        autocast_device_type = "cpu"
    else:
        device = torch.device("cpu")
        autocast_device_type = "cpu"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using device: {device} | Precision: {dtype}")

    model = GPTModel(config.GPT_CONFIG_124M).to(device)

    if hasattr(torch, "compile") and device.type == "cuda" and os.name != "nt":
        model = torch.compile(model)
    else:
     print("Triton missing")

    num_params = sum(p.numel() for p in model.parameters())
    data_jsonl_path = os.path.join(SCRIPT_DIR, "data.jsonl")
    micro_batch_size = 1
    accumulation_steps = 64

    train_loader, val_loader = create_jsonl_dataloaders(
        data_jsonl_path,
        batch_size=micro_batch_size,
        max_length=config.GPT_CONFIG_124M["context_length"],
        stride=config.GPT_CONFIG_124M["context_length"],
        val_ratio=0.1,
    )

    num_epochs = 10
    learning_rate = 5e-4
    max_grad_norm = 1.0
    use_fused_adam = (device.type == "cuda") and ("fused" in torch.optim.AdamW.__doc__)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.1, fused=use_fused_adam
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and dtype == torch.float16) else None

    start_epoch = 0
    checkpoint_path = "gpt_checkpoint_latest.pt"
    history = {"epochs": [], "train_loss": [], "val_loss": [], "tokens_seen": []}

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint '{checkpoint_path}'. Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        history = checkpoint.get("history", history)  
        print(f"Successfully loaded! Resuming training from Epoch {start_epoch + 1}...\n")
    else:
        print("No checkpoint found. Starting training from scratch...\n")

    tokens_per_batch = micro_batch_size * config.GPT_CONFIG_124M["context_length"]
    total_tokens_seen = history["tokens_seen"][-1] if history["tokens_seen"] else start_epoch * len(train_loader) * tokens_per_batch
    best_val_loss = float("inf")
    SAVE_EVERY_N_STEPS = 500
    optimizer_step_count = 0
    training_start_time = time.time()
    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for step, (input_batch, target_batch) in enumerate(pbar):
            step_start_time = time.time()
            loss = calc_loss_batch(input_batch, target_batch, model, device, autocast_device_type, dtype)
            loss = loss / accumulation_steps
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1

                if optimizer_step_count % SAVE_EVERY_N_STEPS == 0:
                    current_loss = loss.item() * accumulation_steps
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, current_loss, history,
                                     filename="gpt_checkpoint_latest.pt")
            step_loss = loss.item() * accumulation_steps
            total_loss += step_loss
            total_tokens_seen += tokens_per_batch
            step_duration = time.time() - step_start_time
            tokens_per_sec = tokens_per_batch / step_duration if step_duration > 0 else 0.0
            step_perplexity = torch.exp(torch.tensor(min(step_loss, 20.0)))
            pbar.set_postfix(
                {
                    "Loss": f"{step_loss:.4f}",
                    "PPL": f"{step_perplexity:.2f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                }
            )
        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device, autocast_device_type, dtype)
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["tokens_seen"].append(total_tokens_seen)
        train_perplexity = torch.exp(torch.tensor(min(avg_train_loss, 20.0)))
        val_perplexity = torch.exp(torch.tensor(min(val_loss, 20.0)))
        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, val_loss, history, filename="gpt_checkpoint_best.pt")
            print(f"New Best Val Loss! Saved gpt_checkpoint_best.pt")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, avg_train_loss, history,
                         filename=f"gpt_checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, step, avg_train_loss, history,
                         filename="gpt_checkpoint_latest.pt")

        plot_loss_curve(history["epochs"], history["train_loss"], history["val_loss"], history["tokens_seen"])
        print(f"=> Loss curve updated: loss_curve.png\n")

    total_training_time = time.time() - training_start_time
    print(f"TRAINING COMPLETE")
    print(f"Total epochs:        {num_epochs}")
    print(f"Total tokens seen:   {total_tokens_seen:,}")
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Final train loss:    {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:      {history['val_loss'][-1]:.4f}")