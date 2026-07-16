# VAE-1D-Domain-Adaptation

Minimal pipeline to train the ResNet-18 classifier (Kermany/RSNA), extract its
features, and train/test the VAE 1D domain-alignment autoencoder (CenterLoss).
This is a trimmed-down slice of `MedicalImagingClassifier`: no plot/report
generation, only the numeric artifacts (`.json`/`.npy`) needed to train and
evaluate the model and the alignment.

## Setup

```bash
pip install -r requirements.txt
```

## Data

1. Download the original datasets and place them under `data/original/[DATASET]` (see `data/README.md`).
2. Process them with the scripts in `scripts/`:

```bash
python scripts/process_kermany.py
python scripts/process_rsna.py
```

This produces `data/processed/kermany/` and `data/processed/rsna/`, each with `train/`, `val/`, `test/`.

## Configuration

There is a single scenario, and it lives entirely in code: the `CONFIG` block
at the top of `main.py`. Edit the constants there to change any setting
(batch size, epochs, `align_weight`, `kl_weight`, `TARGET_DATA_FRACTIONS`,
etc.) — there is no YAML config file. The only thing that varies at runtime
is which dataset is the source, controlled by `--source`.

The default alignment scenario is VAE + CenterLoss with `align_weight=0.9`,
`kl_weight=0.1` (so `mmd_weight = 1 - align_weight - kl_weight = 0.0`), swept
across every fraction in `TARGET_DATA_FRACTIONS`.

## Running

```bash
python main.py --source kermany
python main.py --source rsna
```

Each run trains the ResNet-18 on the source dataset, tests it cross-dataset,
extracts features, computes per-class centroids (KMeans), and trains/tests
the VAE alignment autoencoder against the other dataset for every fraction in
`TARGET_DATA_FRACTIONS`.

Outputs go to `results/resnet18_<source>/`:
- `<dataset>_test_results.json` — classifier accuracy / confusion matrix.
- `autoencoder/<target>/<run>/training_results.json` — VAE training history.
- `autoencoder_test/<target>/<run>/results.json` — classifier accuracy on the aligned features (the alignment test).

Use `--force-retrain` / `--force-reextract` to bypass the weights/features cache.
