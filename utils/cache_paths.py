import os
import json
import time
import errno
import hashlib

TRAIN_HASH_KEYS = [
    "model",
    "layers",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "loss_function",
    "scheduler",
    "use_data_augmentation",
    "data_augmentation",
    "classifier_type",
]

FEATURE_PHASES = ["train", "val", "test"]


def compute_train_hash(config):
    sub = {k: config.get(k) for k in TRAIN_HASH_KEYS}
    canonical = json.dumps(sub, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def cache_root(config=None):
    root = "cache"
    if config is not None:
        root = config.get("cache_root", "cache")
    return os.path.abspath(root)


def weights_cache_dir(root, model, dataset, train_hash):
    return os.path.join(root, "weights", model, dataset, train_hash)


def features_cache_dir(root, model, dataset, train_hash):
    return os.path.join(root, "features", model, dataset, train_hash)


def weights_ready(weights_dir):
    return all(
        os.path.exists(os.path.join(weights_dir, f))
        for f in ["extractor_weights.pt", "classifier_weights.pt", ".complete"]
    )


def features_ready(features_dir, prefix):
    sentinel = os.path.join(features_dir, f".{prefix}.complete")
    if not os.path.exists(sentinel):
        return False
    for phase in FEATURE_PHASES:
        feats = os.path.join(features_dir, f"{prefix}_{phase}_features.npy")
        labels = os.path.join(features_dir, f"{prefix}_{phase}_labels.npy")
        if not (os.path.exists(feats) and os.path.exists(labels)):
            return False
    return True


def mark_weights_complete(weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    open(os.path.join(weights_dir, ".complete"), "w").close()


def mark_features_complete(features_dir, prefix):
    os.makedirs(features_dir, exist_ok=True)
    open(os.path.join(features_dir, f".{prefix}.complete"), "w").close()


def invalidate_features(features_dir):
    if not os.path.isdir(features_dir):
        return
    for f in os.listdir(features_dir):
        if f.startswith(".") and f.endswith(".complete"):
            try:
                os.remove(os.path.join(features_dir, f))
            except OSError:
                pass


class CacheLock:
    def __init__(self, target_dir, name="build", timeout=7200, poll=2.0):
        self.lockfile = os.path.join(target_dir, f".{name}.lock")
        self.timeout = timeout
        self.poll = poll
        self.acquired = False

    def acquire(self):
        os.makedirs(os.path.dirname(self.lockfile), exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.acquired = True
                return True
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if time.monotonic() - start > self.timeout:
                    return False
                time.sleep(self.poll)

    def release(self):
        if self.acquired:
            try:
                os.remove(self.lockfile)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
