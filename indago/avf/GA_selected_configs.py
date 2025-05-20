import subprocess
import csv
import json

# === Define the best configurations manually ===
best_configs = [
    {"model": "blitz", "layers": 4, "learning_rate": 0.001, "under": False, "oversample": 0.0, "seed": 21, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 4, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 22, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 4, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 21, "hidden_layer_size": 128, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 2, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 22, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 1, "learning_rate": 0.001, "under": True,  "oversample": 1.0, "seed": 21, "hidden_layer_size": 128, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 4, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 21, "hidden_layer_size": 32, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 3, "learning_rate": 0.001, "under": False, "oversample": 1.0, "seed": 22, "hidden_layer_size": 32, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 3, "learning_rate": 0.001, "under": True,  "oversample": 0.5, "seed": 23, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 3, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 23, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 2, "learning_rate": 0.001, "under": True,  "oversample": 1.0, "seed": 22, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 3, "learning_rate": 0.001, "under": False, "oversample": 1.0, "seed": 22, "hidden_layer_size": 64, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 2, "learning_rate": 0.001, "under": False, "oversample": 1.0, "seed": 23, "hidden_layer_size": 32, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
    {"model": "blitz", "layers": 1, "learning_rate": 0.001, "under": True,  "oversample": 0.0, "seed": 20, "hidden_layer_size": 128, "test_split": 0.1, "batch_size": 128, "augment": True, "weight_loss": False},
]

# === Output CSV setup ===
output_file = "selected_configs_results.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        'model', 'layers', 'learning_rate', 'under', 'oversample', 'seed',
        'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss',
        'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'test_loss',
        'ga_mean', 'ga_std', 'ga_min', 'ga_max'
    ])

# === Run each configuration ===
for i, config in enumerate(best_configs, 1):
    print(f"\n[{i}/{len(best_configs)}] Running config: {config}")

    # === Train model ===
    train_cmd = [
        "python", "-m", "indago.avf.train",
        "--algo", "her", "--env-name", "park", "--env-id", "parking-v0",
        "--exp-id", "1", "--test-split", str(config["test_split"]), "--avf-policy", "bnn",
        "--training-progress-filter", "50", "--oversample", str(config["oversample"]),
        "--n-epochs", "40", "--learning-rate", str(config["learning_rate"]), "--batch-size", str(config["batch_size"]),
        "--patience", "4", "--hidden-layer-size", str(config["hidden_layer_size"]), "--layers", str(config["layers"]),
        "--seed", str(config["seed"]),
        "--heldout-test-file", "heldout-set-seed-21-0.2-split-5-filter-cls.npz",
        "--model", config["model"]
    ]
    if config["under"]: train_cmd.append("--under")
    if config["augment"]: train_cmd.append("--augment")
    if config["weight_loss"]: train_cmd.append("--weight-loss")

    result = None
    process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        if "FINAL_RESULTS:" in line:
            json_part = line.strip().split("FINAL_RESULTS:")[1]
            result = json.loads(json_part)
    process.wait()

    # === Run GA ===
    ga_cmd = [
        "python", "-m", "indago.experiments",
        "--algo", "her", "--exp-id", "1", "--env-name", "park", "--env-id", "parking-v0",
        "--avf-train-policy", "bnn", "--avf-test-policy", "ga_saliency_rnd",
        "--failure-prob-dist", "--num-episodes", "25", "--num-runs-each-env-config", "1",
        "--training-progress-filter", "50", "--layers", str(config["layers"]),
        "--budget", "3", "--num-runs-experiments", "1", "--oversample", str(config["oversample"])
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
            config["model"], config["layers"], config["learning_rate"], config["under"], config["oversample"],
            config["seed"], config["hidden_layer_size"], config["test_split"], config["batch_size"],
            config["augment"], config["weight_loss"],
            result.get("accuracy") if result else None,
            result.get("precision") if result else None,
            result.get("recall") if result else None,
            result.get("f_measure") if result else None,
            result.get("auc_roc") if result else None,
            result.get("test_loss") if result else None,
            ga_mean, ga_std, ga_min, ga_max
        ])

print("✅ Selected configs execution complete.")
