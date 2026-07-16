import os
import time
import torch
from utils.metrics import get_memory_usage, get_model_size, get_parameters_count, calculate_flops, measure_inference_time

from tqdm import tqdm
from tempfile import TemporaryDirectory

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=100,
    early_stopping_patience=10,
    verbose=False,
):
    model = model.to(device)

    since = time.time()

    metrics = {
        "total_time_seconds": 0,
        "epoch_times": [],
        "batch_times": {"train": [], "val": []},
        "throughput": {"train": [], "val": []},  # samples/second
        "memory_usage": [],
        "model_info": {
            **get_parameters_count(model),
            "model_size_mb": get_model_size(model),
        },
    }

    initial_memory = get_memory_usage()
    metrics["initial_memory"] = initial_memory

    with TemporaryDirectory() as tempdir:
        best_model_params_path = tempdir
        if verbose:
            print(f"Best weights saved to {best_model_params_path}")
        best_val_loss = float("inf")
        patience_counter = 0
        early_stop = False

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            if early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break
            epoch_start = time.time()
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
            if verbose:
                print("-" * 10)

            for phase in ["train", "val"]:
                phase_start = time.time()
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples_processed = 0
                batch_times = []
                pbar = tqdm(
                    dataloaders[phase],
                    desc=f"{phase.capitalize():5s}",
                    unit="batch",
                    leave=False,
                    disable=not verbose,
                )

                for inputs, labels in pbar:
                    batch_start = time.time()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    total_samples_processed += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data).float() / inputs.size(
                        0
                    )
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                phase_time = time.time() - phase_start
                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.float() / total_samples_processed

                # Save the phase's computational metrics
                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )

                metrics["batch_times"][phase].append(
                    {
                        "epoch": epoch,
                        "avg_batch_time_seconds": avg_batch_time,
                        "total_batches": len(batch_times),
                        "total_time_seconds": phase_time,
                    }
                )

                metrics["throughput"][phase].append(
                    {
                        "epoch": epoch,
                        "total_samples": total_samples_processed,
                        "total_time_seconds": phase_time,
                    }
                )

                if verbose:
                    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                if verbose:
                    print(f"{phase} Time: {phase_time:.2f}s")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())
                    current_val_loss = epoch_loss  # Save to use in the scheduler

                if verbose:
                    print(
                        f"{phase.capitalize():5s} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
                    )

                if phase == "val":
                    if epoch_loss < best_val_loss:
                        if verbose:
                            print(f"Val loss improved from {best_val_loss:.4f} to {epoch_loss:.4f}")
                        best_val_loss = epoch_loss
                        patience_counter = 0
                        model.save_weights(best_model_params_path)
                    elif early_stopping_patience is not None:
                        patience_counter += 1
                        if verbose:
                            print(f"Val loss did not improve from {best_val_loss:.4f} - Patience: {patience_counter}")
                        if patience_counter >= early_stopping_patience:
                            early_stop = True
                            if verbose:
                                print("Early stopping triggered")
                            break

            if early_stop:
                break

            if scheduler is not None:
                # ReduceLROnPlateau needs the validation metric
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_loss)
                else:
                    scheduler.step()

                if verbose:
                    current_lr = scheduler.optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - New Learning Rate: {current_lr:.6f}"
                    )

            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append({"epoch": epoch, "time_seconds": epoch_time})

            epoch_memory = get_memory_usage()
            metrics["memory_usage"].append({"epoch": epoch, **epoch_memory})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if verbose:
                print()

        time_elapsed = time.time() - since
        metrics["total_time_seconds"] = time_elapsed

        final_memory = get_memory_usage()
        metrics["final_memory"] = final_memory

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if verbose:
            print(
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        if verbose:
            print(f"Best loss [val]: {best_val_loss:.4f}")

        model.load_weights(best_model_params_path)

        # Add inference and FLOPs metrics at the end
        print("Computing inference and complexity metrics...")
        input_shape = next(iter(dataloaders['train']))[0].shape
        metrics["flops"] = calculate_flops(model, input_shape)
        metrics["inference_metrics"] = measure_inference_time(model, input_shape)

    return model, history, metrics


def train_model_no_early_stopping(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=30,
    verbose=False,
):
    """
    Trains the model for exactly num_epochs epochs, without early stopping.
    Implements the training methodology from Kundu et al.

    Args:
        model: Model to be trained
        pretrained_model: Name of the pretrained backbone
        dataloaders: Dict with 'train' and 'val' DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Exact number of epochs to train for (default: 30)
        verbose: If True, prints detailed progress

    Returns:
        model: Trained model (state of the last epoch)
        history: Dict with loss and accuracy history
        metrics: Dict with computational metrics
    """
    if verbose:
        print(f"Moving model to device: {device}...")
    model = model.to(device)
    if verbose:
        print(f"Model moved to {device} successfully!")

    since = time.time()

    metrics = {
        "total_time_seconds": 0,
        "epoch_times": [],
        "batch_times": {"train": [], "val": []},
        "throughput": {"train": [], "val": []},
        "memory_usage": [],
        "model_info": {
            **get_parameters_count(model),
            "model_size_mb": get_model_size(model),
        },
    }

    initial_memory = get_memory_usage()
    metrics["initial_memory"] = initial_memory

    with TemporaryDirectory() as tempdir:
        best_model_params_path = tempdir
        best_val_loss = float("inf")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        if verbose:
            print("Starting epoch loop...")

        for epoch in range(num_epochs):
            epoch_start = time.time()
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print("-" * 10)

            for phase in ["train", "val"]:
                phase_start = time.time()
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples_processed = 0
                batch_times = []

                if verbose and epoch == 0:
                    print(f"Loading first {phase} batch...")

                pbar = tqdm(
                    dataloaders[phase],
                    desc=f"{phase.capitalize():5s}",
                    unit="batch",
                    leave=False,
                    disable=not verbose,
                )

                for inputs, labels in pbar:
                    batch_start = time.time()
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    if phase == "train":
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                    total_samples_processed += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    batch_acc = torch.sum(preds == labels.data).float() / inputs.size(0)
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                    )

                phase_time = time.time() - phase_start
                epoch_loss = running_loss / total_samples_processed
                epoch_acc = running_corrects.float() / total_samples_processed

                avg_batch_time = (
                    sum(batch_times) / len(batch_times) if batch_times else 0
                )

                metrics["batch_times"][phase].append(
                    {
                        "epoch": epoch,
                        "avg_batch_time_seconds": avg_batch_time,
                        "total_batches": len(batch_times),
                        "total_time_seconds": phase_time,
                    }
                )

                metrics["throughput"][phase].append(
                    {
                        "epoch": epoch,
                        "total_samples": total_samples_processed,
                        "total_time_seconds": phase_time,
                    }
                )

                if verbose:
                    print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                    print(f"{phase} Time: {phase_time:.2f}s")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())
                    current_val_loss = epoch_loss

                if phase == "val":
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        model.save_weights(best_model_params_path)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_val_loss)
                else:
                    scheduler.step()

                if verbose:
                    current_lr = scheduler.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}/{num_epochs} - Learning Rate: {current_lr:.6f}")

            epoch_time = time.time() - epoch_start
            metrics["epoch_times"].append({"epoch": epoch, "time_seconds": epoch_time})

            epoch_memory = get_memory_usage()
            metrics["memory_usage"].append({"epoch": epoch, **epoch_memory})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if verbose:
                print()

        time_elapsed = time.time() - since
        metrics["total_time_seconds"] = time_elapsed

        final_memory = get_memory_usage()
        metrics["final_memory"] = final_memory

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if verbose:
            print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            print(f"Best loss [val]: {best_val_loss:.4f}")

        # Load the best model (unlike the original, which uses the last one)
        # We keep this behavior for consistency with evaluation
        model.load_weights(best_model_params_path)

        print("Computing inference and complexity metrics...")
        input_shape = next(iter(dataloaders['train']))[0].shape
        metrics["flops"] = calculate_flops(model, input_shape)
        metrics["inference_metrics"] = measure_inference_time(model, input_shape)

    return model, history, metrics
