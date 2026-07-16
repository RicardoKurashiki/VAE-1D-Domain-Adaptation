import os
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

from models.autoencoder import AutoEncoder
from models.classification_model import ClassificationModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _load_ae(ae_weights_path, architecture, input_dim):
    model = AutoEncoder(architecture, input_dim=input_dim).model.to(device)
    model.load_state_dict(torch.load(ae_weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_classifier(model_path, num_classes, backbone, classifier_type):
    classification_model = ClassificationModel(
        num_classes=num_classes,
        backbone=backbone,
        classifier_type=classifier_type,
    )
    classification_model.load_weights(model_path)
    classifier = classification_model.model.classifier.to(device)
    classifier.eval()
    return classifier


def _align_features(ae_model, features, architecture):
    with torch.no_grad():
        features_t = torch.tensor(features, dtype=torch.float32).to(device)
        if architecture == "vae":
            x_recon, z, mean, log_var = ae_model(features_t)
        else:
            x_recon, z = ae_model(features_t)
    recon = x_recon.cpu().numpy()
    errors = np.mean((features - recon) ** 2, axis=1)
    return recon, errors


def _classify(classifier, features):
    with torch.no_grad():
        features_t = torch.tensor(features, dtype=torch.float32).to(device)
        logits = classifier(features_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


def run(model_path, ae_run_dir, target_dataset,
        pretrained_model, num_classes, classifier_type,
        architecture=None,
        output_path=None, output_path_base=None,
        weights_dir=None, features_dir=None):
    run_name = os.path.basename(ae_run_dir)

    if weights_dir is None:
        weights_dir = model_path
    if features_dir is None:
        features_dir = os.path.join(model_path, "features")

    if architecture is None:
        architecture = run_name.split("_")[0]

    if output_path is None:
        output_path = os.path.join(output_path_base, target_dataset, run_name)

    os.makedirs(output_path, exist_ok=True)

    features = np.load(os.path.join(features_dir, f"{target_dataset}_test_features.npy"))
    labels = np.load(os.path.join(features_dir, f"{target_dataset}_test_labels.npy"))

    ae_model = _load_ae(os.path.join(ae_run_dir, "weights.pt"), architecture, input_dim=features.shape[1])
    classifier = _load_classifier(weights_dir, num_classes, pretrained_model, classifier_type)

    aligned, recon_errors = _align_features(ae_model, features, architecture)

    np.save(os.path.join(output_path, "aligned_features.npy"), aligned)
    np.save(os.path.join(output_path, "aligned_labels.npy"), labels)
    np.save(os.path.join(output_path, "reconstruction_errors.npy"), recon_errors)

    preds = _classify(classifier, aligned)
    np.save(os.path.join(output_path, "aligned_preds.npy"), preds)

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)

    n_train_samples = None
    training_results_path = os.path.join(ae_run_dir, "training_results.json")
    if os.path.isfile(training_results_path):
        with open(training_results_path) as f:
            n_train_samples = json.load(f).get("config", {}).get("n_train_samples")

    results = {
        "run": run_name,
        "target_dataset": target_dataset,
        "architecture": architecture,
        "n_train_samples": n_train_samples,
        "accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{run_name}] accuracy={report['accuracy']:.4f}")
    return results


def sweep_test(model_path, ae_sweep_path, target_datasets,
               pretrained_model, num_classes, classifier_type, output_path,
               weights_dir=None, features_dir=None):
    if weights_dir is None:
        weights_dir = model_path
    if features_dir is None:
        features_dir = os.path.join(model_path, "features")

    all_results = []

    for target in target_datasets:
        target_sweep_dir = os.path.join(ae_sweep_path, target)
        if not os.path.isdir(target_sweep_dir):
            print(f"Skipping {target}: no sweep dir at {target_sweep_dir}")
            continue

        for run_name in sorted(os.listdir(target_sweep_dir)):
            ae_run_dir = os.path.join(target_sweep_dir, run_name)
            weights_path = os.path.join(ae_run_dir, "weights.pt")
            if not os.path.isfile(weights_path):
                continue

            architecture = run_name.split("_")[0]
            run_output = os.path.join(output_path, target, run_name)

            result = run(
                model_path=model_path,
                ae_run_dir=ae_run_dir,
                target_dataset=target,
                architecture=architecture,
                pretrained_model=pretrained_model,
                num_classes=num_classes,
                classifier_type=classifier_type,
                output_path=run_output,
                weights_dir=weights_dir,
                features_dir=features_dir,
            )
            all_results.append(result)

    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
