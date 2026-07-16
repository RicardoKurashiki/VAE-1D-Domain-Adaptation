import os
import time
import json
import threading
import numpy as np
import torch
import psutil
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import AutoEncoder
from loss import CenterLoss

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _compute_kl(mean, log_var):
    return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())


def _assign_centroids_to_classes(centroid_labels, num_classes):
    assignment = np.arange(num_classes)
    for cls in range(num_classes):
        matches = np.where(centroid_labels == cls)[0]
        if len(matches) > 0:
            assignment[cls] = matches[0]
    return assignment


def _build_align_loss(align_fn, input_dim, num_classes, centroids=None, assignment=None, margin=1.0):
    if align_fn in ("none", None):
        return None

    initial_centers = None
    if centroids is not None:
        if assignment is not None:
            ordered = np.array([centroids[assignment[cls]] for cls in range(num_classes)])
        else:
            ordered = centroids[:num_classes]
        initial_centers = torch.tensor(ordered, dtype=torch.float32)

    if align_fn == "center":
        return CenterLoss(
            num_classes=num_classes, feat_dim=input_dim,
            use_gpu=(device != "cpu"), initial_centers=initial_centers,
        ).to(device)
    raise ValueError(f"Unknown align_fn: {align_fn}")

def _make_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32).to(device),
        torch.tensor(labels, dtype=torch.long).to(device),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / (1024 ** 2), 4)


def _compute_flops(model, input_dim, architecture="simple", n_classes=2):
    try:
        import copy
        from thop import profile
        # thop.profile registers float64 total_ops/total_params buffers on the
        # model it profiles and never removes them - profile a throwaway copy so
        # those buffers don't end up in the checkpoint we save/load afterwards
        # (MPS can't rebuild float64 tensors on load).
        dummy = torch.zeros(1, input_dim).to(device)
        flops, _ = profile(copy.deepcopy(model), inputs=(dummy,), verbose=False)
        return int(flops)
    except Exception:
        return None


def _forward(model, architecture, features, labels, n_classes):
    if architecture == "vae":
        x_recon, z, mean, log_var = model(features)
        kl = _compute_kl(mean, log_var)
    else:
        x_recon, z = model(features)
        kl = torch.tensor(0.0)
    return x_recon, z, kl


def train(model, train_loader, val_loader, epochs, lr, save_path,
          architecture, align_loss_fn=None, align_weight=0.0, kl_weight=0.1,
          early_stopping_patience=10, n_classes=2):
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = {
        'train_total': [],
        'train_align': [],
        'train_kl': [],
        'epoch_time': [],
    }

    uses_kl = architecture == "vae"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ep_total = ep_align = ep_kl = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()

            x_recon, z, kl = _forward(model, architecture, features, labels, n_classes)

            loss = torch.zeros((), device=device)
            if uses_kl:
                loss = loss + kl_weight * kl

            align_val = torch.tensor(0.0)
            if align_loss_fn is not None and align_weight > 0:
                align_val = align_loss_fn(x_recon, labels)
                loss = loss + align_weight * align_val

            loss.backward()
            optimizer.step()

            n = len(train_loader)
            ep_total += loss.item()
            ep_align += align_val.item() if isinstance(align_val, torch.Tensor) else align_val
            ep_kl += kl.item() if isinstance(kl, torch.Tensor) else kl

        n = len(train_loader)
        history['train_total'].append(ep_total / n)
        history['train_align'].append(ep_align / n)
        history['train_kl'].append(ep_kl / n)

        model.eval()
        val_align = val_kl = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                x_recon, z, kl = _forward(model, architecture, features, labels, n_classes)
                if uses_kl:
                    val_kl += kl.item()
                if align_loss_fn is not None and align_weight > 0:
                    val_align += align_loss_fn(x_recon, labels).item()

        n_val = len(val_loader)
        val_align /= n_val
        val_kl /= n_val
        val_total = align_weight * val_align + (kl_weight * val_kl if uses_kl else 0.0)

        history.setdefault('val_align', []).append(val_align)
        history.setdefault('val_kl', []).append(val_kl)
        history.setdefault('val_total', []).append(val_total)

        epoch_time = time.time() - t0
        history['epoch_time'].append(round(epoch_time, 3))

        print(
            f"[{architecture}] Epoch {epoch}/{epochs} "
            f"| train={ep_total/n:.6f}"
            + (f" align={ep_align/n:.6f}" if align_loss_fn is not None else "")
            + (f" kl={ep_kl/n:.6f}" if uses_kl else "")
            + f" | val_total={val_total:.6f}"
            + (f" val_align={val_align:.6f}" if align_loss_fn is not None else "")
            + (f" val_kl={val_kl:.6f}" if uses_kl else "")
            + f" | {epoch_time:.1f}s"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                break

    print(f"Best val_total={best_val_loss:.6f} -> {save_path}")
    return history


def run(latent_path, centroid_path, source_dataset="kermany", target_dataset="rsna", architecture="vae",
        align_fn="none", align_weight=0.0, align_margin=1.0, epochs=100, lr=1e-3, batch_size=64,
        kl_weight=0.01, early_stopping_patience=10, output_path=None,
        target_data_fraction=1.0, random_state=42):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)

    uses_kl = architecture == "vae"

    train_features = np.load(os.path.join(latent_path, f"{target_dataset}_train_features.npy"))
    train_labels = np.load(os.path.join(latent_path, f"{target_dataset}_train_labels.npy"))

    if target_data_fraction < 1.0:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(
            np.arange(len(train_features)),
            train_size=target_data_fraction,
            stratify=train_labels,
            random_state=random_state,
        )
        train_features = train_features[idx]
        train_labels = train_labels[idx]
    val_features = np.load(os.path.join(latent_path, f"{target_dataset}_val_features.npy"))
    val_labels = np.load(os.path.join(latent_path, f"{target_dataset}_val_labels.npy"))

    centroids = np.load(os.path.join(centroid_path, "cluster_centers.npy"))
    centroid_labels = np.load(os.path.join(centroid_path, "centroid_labels.npy"))

    num_classes = int(train_labels.max()) + 1
    input_dim = train_features.shape[1]
    latent_dim = 64

    assignment = _assign_centroids_to_classes(centroid_labels, num_classes)
    print(f"Centroid assignment (class -> centroid index): {assignment}")

    train_loader = _make_loader(train_features, train_labels, batch_size, shuffle=True)
    val_loader = _make_loader(val_features, val_labels, batch_size, shuffle=False)

    model = AutoEncoder(architecture, input_dim=input_dim, n_classes=num_classes).model.to(device)
    align_loss_fn = _build_align_loss(align_fn, input_dim, num_classes, centroids, assignment, margin=align_margin)

    kl_suffix = f"_kl{kl_weight:.3f}" if architecture == "vae" else ""
    frac_suffix = f"_frac{target_data_fraction:.2f}"
    run_dir = os.path.join(output_path, f"{architecture}_{align_fn}_{align_weight:.2f}{kl_suffix}{frac_suffix}")
    save_path = os.path.join(run_dir, "weights.pt")

    print(f"\n{'='*60}")
    print(f"target={target_dataset} | architecture={architecture} | align_fn={align_fn} | align_weight={align_weight}" +
          (f" | kl_weight={kl_weight}" if architecture == "vae" else ""))
    print(f"{'='*60}")

    n_params = _count_parameters(model)
    size_mb = _model_size_mb(model)
    flops = _compute_flops(model, input_dim, architecture=architecture, n_classes=num_classes)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    _proc = psutil.Process()
    peak_cpu_memory_bytes = [_proc.memory_info().rss]
    stop_monitor = threading.Event()

    def _monitor_cpu():
        while not stop_monitor.is_set():
            peak_cpu_memory_bytes.append(_proc.memory_info().rss)
            stop_monitor.wait(0.5)

    monitor_thread = threading.Thread(target=_monitor_cpu, daemon=True)
    monitor_thread.start()

    t_start = time.time()
    history = train(
        model, train_loader, val_loader, epochs, lr, save_path,
        architecture=architecture,
        align_loss_fn=align_loss_fn,
        align_weight=align_weight, kl_weight=kl_weight,
        early_stopping_patience=early_stopping_patience,
        n_classes=num_classes,
    )
    total_time = time.time() - t_start

    stop_monitor.set()
    monitor_thread.join(timeout=2)

    peak_gpu_mb = None
    if device == "cuda":
        peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    peak_cpu_memory_mb = round(max(peak_cpu_memory_bytes) / (1024 ** 2), 2)

    metrics = {
        'n_trainable_parameters': n_params,
        'model_size_mb': size_mb,
        'flops': flops,
        'total_training_time_s': round(total_time, 2),
        'avg_epoch_time_s': round(float(np.mean(history['epoch_time'])), 3),
        'peak_gpu_memory_mb': peak_gpu_mb,
        'peak_cpu_memory_mb': peak_cpu_memory_mb,
        'device': device,
    }

    run_config = {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'architecture': architecture,
        'align_fn': align_fn,
        'align_weight': align_weight,
        'kl_weight': kl_weight if uses_kl else 0.0,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'target_data_fraction': target_data_fraction,
        'n_train_samples': len(train_features),
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'training_results.json'), 'w') as f:
        json.dump({'config': run_config, 'metrics': metrics, 'history': history}, f, indent=2)

    print(f"Saved results -> {run_dir}")
    return run_dir


def sweep(latent_path, centroid_path, source_dataset="kermany", target_datasets=("rsna",),
          architectures=("vae",), align_fns=("center",),
          align_weights=(0.9,),
          align_margin=1.0,
          kl_weights=(0.1,),
          target_data_fractions=(1.0,),
          epochs=100, lr=1e-3, batch_size=64,
          early_stopping_patience=10, output_path=None,
          test_fn=None, test_kwargs=None):
    for target in target_datasets:
        for fraction in target_data_fractions:
            for arch in architectures:
                for align_fn in align_fns:
                    weights_to_run = [0.0] if align_fn in ("none", None) else align_weights
                    kl_weights_to_run = kl_weights if arch == "vae" else [0.0]
                    for weight in weights_to_run:
                        for kl_weight in kl_weights_to_run:
                            run_dir = run(
                                latent_path=latent_path,
                                centroid_path=centroid_path,
                                source_dataset=source_dataset,
                                target_dataset=target,
                                architecture=arch,
                                align_fn=align_fn,
                                align_weight=weight,
                                align_margin=align_margin,
                                epochs=epochs,
                                lr=lr,
                                batch_size=batch_size,
                                kl_weight=kl_weight,
                                early_stopping_patience=early_stopping_patience,
                                output_path=os.path.join(output_path, target),
                                target_data_fraction=fraction,
                            )
                            if run_dir is not None and test_fn is not None and test_kwargs is not None:
                                test_fn(ae_run_dir=run_dir, target_dataset=target, **test_kwargs)
