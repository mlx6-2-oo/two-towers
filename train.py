import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import get_datasets
from model import DualTowerModel, margin
from utils import get_device


def train():
    num_epochs = 10

    torch.manual_seed(7)

    device = get_device()
    batch_size = 512 if device.type in ["cuda"] else 32
    num_workers = 2 if device.type in ["cpu", "cuda"] else 0
    persistent_workers = True if num_workers > 0 else False

    # Initialize model
    model = DualTowerModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Get datasets and dataloaders
    print("Loading training data...")
    triple_dataset = get_datasets("train")
    triple_dataloader = DataLoader(
        triple_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=triple_dataset.collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    print("Loading validation data...")
    val_triple_dataset = get_datasets("validation")
    val_triple_dataloader = DataLoader(
        val_triple_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=triple_dataset.collate_fn,
        drop_last=True,
    )

    wandb.init(
        project="two-towers",
        config={
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "margin": margin,
        },
    )

    # Training loop
    for epoch in range(num_epochs):
        batch_losses = []

        train_loop = tqdm(
            triple_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(triple_dataloader),
        )

        for batch_idx, (query_batch, document_batch, neg_document_batch) in enumerate(
            train_loop
        ):
            model.train()
            optimizer.zero_grad()

            # Compute loss
            loss, unclamped_loss, relevant_distance, irrelevant_distance = (
                model.compute_loss(query_batch, document_batch, neg_document_batch)
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "batch": batch_idx + epoch * len(triple_dataloader),
                    "unclamped_loss": unclamped_loss.item(),
                    "relevant_distance": relevant_distance.item(),
                    "irrelevant_distance": irrelevant_distance.item(),
                }
            )

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    # validate with just the first batch
                    model.eval()
                    val_query_batch, val_documents_batch, val_neg_documents_batch = (
                        next(iter(val_triple_dataloader))
                    )
                    val_loss, unclamped_loss, relevant_distance, irrelevant_distance = (
                        model.compute_loss(
                            val_query_batch,
                            val_documents_batch,
                            val_neg_documents_batch,
                        )
                    )
                    wandb.log(
                        {
                            "batch_val_loss": val_loss.item(),
                            "batch_val_unclamped_loss": unclamped_loss.item(),
                            "batch_val_relevant_distance": relevant_distance.item(),
                            "batch_val_irrelevant_distance": irrelevant_distance.item(),
                        }
                    )

        train_loss = sum(batch_losses) / len(batch_losses)

        # save weights and upload to wandb
        torch.save(model.state_dict(), f"two_towers_{epoch}.pth")
        wandb.save(f"two_towers_{epoch}.pth")

        # Validation loop
        val_losses = []
        with torch.no_grad():
            model.eval()

            val_loop = tqdm(
                val_triple_dataloader,
                desc="Validation",
                total=len(val_triple_dataloader),
            )

            for batch_idx, (
                val_query_batch,
                val_documents_batch,
                val_neg_documents_batch,
            ) in enumerate(val_loop):
                if batch_idx > 10:
                    break
                # Compute validation loss
                val_loss, unclamped_loss, relevant_distance, irrelevant_distance = (
                    model.compute_loss(
                        val_query_batch, val_documents_batch, val_neg_documents_batch
                    )
                )

                val_losses.append(val_loss.item())
                val_loop.set_postfix(loss=f"{val_loss.item():.4f}")

        val_loss = sum(val_losses) / len(val_losses)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
