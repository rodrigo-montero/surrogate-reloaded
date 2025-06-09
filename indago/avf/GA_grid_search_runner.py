"""
Purpose of this file is to run a grid search directly on the Genetic Algorithm.
It first trains the model, then runs the GA.
"""
import subprocess
import itertools
import csv
import json
import time

# === Parameter grid setup ===
LAYERS_LIST = [2]
LR_LIST = [1e-3]
UNDER = [False]
OVERSAMPLE_LIST = [0.0]
SEED_LIST = [-1]
HIDDEN_LAYER_SIZE = [64]
TEST_SPLIT = [0.1]
BATCH_SIZE = [128]
MODEL = ["blitz"]
AUGMENT = [True]
WEIGHT_LOSS = [True]

param_grid = list(itertools.product(
    MODEL, LAYERS_LIST, LR_LIST, UNDER, OVERSAMPLE_LIST,
    SEED_LIST, HIDDEN_LAYER_SIZE, TEST_SPLIT, BATCH_SIZE, AUGMENT, WEIGHT_LOSS
))

# === Output setup ===
output_file = "grid_search_with_ga_baseline.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        'model', 'layers', 'learning_rate', 'under', 'oversample', 'seed',
        'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss',
        'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'test_loss',
        'ga_mean', 'ga_std', 'ga_min', 'ga_max'
    ])

# === Run each configuration ===
for i, (model, layers, lr, under, oversample, seed, hidden, ts, bs, augment, wl) in enumerate(param_grid, 1):
    print(f"\n[{i}/{len(param_grid)}] Running config: {model}, layers={layers}, seed={seed}, oversample={oversample}")

    # === Train model ===
    train_cmd = [
        "python", "-m", "indago.avf.train",
        "--algo", "her", "--env-name", "park", "--env-id", "parking-v0",
        "--exp-id", "1", "--test-split", str(ts), "--avf-policy", "bnn",
        "--training-progress-filter", "50", "--oversample", str(oversample),
        "--n-epochs", "40", "--learning-rate", str(lr), "--batch-size", str(bs),
        "--patience", "4", "--hidden-layer-size", str(hidden), "--layers", str(layers),
        "--seed", str(seed),
        "--heldout-test-file", "heldout-set-seed-21-0.2-split-5-filter-cls.npz",
        "--model", str(model)
    ]
    if under: train_cmd.append("--under")
    if augment: train_cmd.append("--augment")
    if wl: train_cmd.append("--weight-loss")

    result = None
    process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        if "FINAL_RESULTS:" in line:
            json_part = line.strip().split("FINAL_RESULTS:")[1]
            result = json.loads(json_part)
    process.wait()

    # === Run GA ===
    for i in range(5): # This range should do the same as the 'num-runs-experiments' variable
        ga_cmd = [
            "python", "-m", "indago.experiments",
            "--algo", "her", "--exp-id", "1", "--env-name", "park", "--env-id", "parking-v0",
            "--avf-train-policy", "bnn", "--avf-test-policy", "ga_saliency_rnd",
            "--failure-prob-dist", "--num-episodes", "50", "--num-runs-each-env-config", "1",
            "--training-progress-filter", "50", "--layers", str(layers),
            "--budget", "3", "--num-runs-experiments", "1", "--oversample", str(oversample)
        ]
        ga_mean = ga_std = ga_min = ga_max = None
        process = subprocess.Popen(ga_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            if "Failure probabilities:" in line:
                parts = line.strip().split("Mean:")[1].split(",")
                ga_mean = float(parts[0].strip())
                ga_std = float(parts[1].split(":")[1].strip())
                ga_min = float(parts[2].split(":")[1].strip())
                ga_max = float(parts[3].split(":")[1].strip())
        process.wait()

        # === Write to CSV ===
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model, layers, lr, under, oversample, seed, hidden, ts, bs, augment, wl,
                result.get("accuracy") if result else None,
                result.get("precision") if result else None,
                result.get("recall") if result else None,
                result.get("f_measure") if result else None,
                result.get("auc_roc") if result else None,
                result.get("test_loss") if result else None,
                ga_mean, ga_std, ga_min, ga_max
            ])

print("Grid + GA search complete.")
