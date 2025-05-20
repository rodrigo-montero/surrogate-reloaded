"""
This file is responsible for running the grid search
Write: 'python /indago/avf/grid_search_runner.py' in terminal to run
Will create a new file in the root folder with the results
"""
import subprocess
import itertools
import csv
import json
import time

start_time = time.time()


# Define all the parameters that will be searched
LAYERS_LIST = [1, 2, 3, 4]                                    # Mainly 3 layers
LR_LIST = [1e-3]
UNDER = [True, False]                                         # Keep to False
OVERSAMPLE_LIST = [0.0, 0.5, 1.0]                       # Maybe 1.0
SEED_LIST = [20, 21, 22, 23, 24]
HIDDEN_LAYER_SIZE = [32, 64, 128]                           # Mainly 64
TEST_SPLIT = [0.1]                                      # Doesn't really matter, but 0.2 is better
BATCH_SIZE = [128]                                      # Mainly 128
MODEL = ["blitz"]                                       # Use blitz
AUGMENT = [True, False]                                        # Mainly True but can't confirm
WEIGHT_LOSS = [True, False]
"""
LAYERS_LIST = [3, 4]                                    # Mainly 3 layers
LR_LIST = [5e-3, 1e-3]
UNDER = [False]                                         # Keep to False
OVERSAMPLE_LIST = [0.0, 0.5, 1.0]                       # Maybe 1.0
SEED_LIST = [21]
HIDDEN_LAYER_SIZE = [64, 128]                           # Mainly 64
TEST_SPLIT = [0.1, 0.2]                                      # Doesn't really matter, but 0.2 is better
BATCH_SIZE = [128]                                     # Mainly 128
MODEL = ["blitz"]                                       # Use blitz
AUGMENT = [True]                                        # Mainly True but can't confirm
WEIGHT_LOSS = [True]

LAYERS_LIST = [2, 3]                                    # Mainly 3 layers
LR_LIST = [1e-3]
UNDER = [False]                                         # Keep to False
OVERSAMPLE_LIST = [0.0, 0.5, 1.0]                       # Maybe 1.0
SEED_LIST = [21]
HIDDEN_LAYER_SIZE = [64]                                # Mainly 64
TEST_SPLIT = [0.2]                                      
BATCH_SIZE = [128]                                      # Mainly 128
MODEL = ["mc-dropout", "blitz"]                                       # Use blitz
AUGMENT = [True, False]                                 # Mainly True but can't confirm
WEIGHT_LOSS = [True]


LAYERS_LIST = [1, 2, 3]                                 # Mainly 3 layers
LR_LIST = [1e-4]
UNDER = [True, False]                                   # Keep to False
OVERSAMPLE_LIST = [0.0, 0.5, 1.0]                       # Maybe 1.0
SEED_LIST = [21]
HIDDEN_LAYER_SIZE = [16, 32, 64]                        # Mainly 64
TEST_SPLIT = [0.2, 0.3]                                 # Doesn't really matter, but 0.2 is better
BATCH_SIZE = [128, 256]                                 # Mainly 128
MODEL = ["blitz"] # "mc-dropout", 
AUGMENT = [True, False]                                 # Mainly True but can't confirm
WEIGHT_LOSS = [True]


LAYERS_LIST = [1]
LR_LIST = [1e-4]
UNDER = [False]
OVERSAMPLE_LIST = [0.0]
SEED_LIST = [21]  # Optional: different seeds for robustness
HIDDEN_LAYER_SIZE = [16]
TEST_SPLIT = [0.3]
BATCH_SIZE = [256]
MODEL = ["blitz"]
AUGMENT = [True, False]

"""

param_grid = list(itertools.product(MODEL, LAYERS_LIST, LR_LIST, UNDER, OVERSAMPLE_LIST, SEED_LIST, HIDDEN_LAYER_SIZE, TEST_SPLIT, BATCH_SIZE, AUGMENT, WEIGHT_LOSS))
total_runs = len(param_grid)

output_file = "grid_search_results.csv"

# Write CSV headers
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'layers', 'learning_rate', 'under', 'oversample', 'seed', 'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss', 'accuracy', 'precision', 'recall', 'f_measure', 'auc_roc', 'test_loss', 'best_epochs'])

# Loop through combinations
for run_idx, (model, layers, lr, under, oversample, seed, hidden, ts, bs, augment, wl) in enumerate(param_grid, 1):
    cmd = [
        "python", "-m", "indago.avf.train",
        "--algo", "her",
        "--env-name", "park",
        "--env-id", "parking-v0",
        "--exp-id", "1",
        "--test-split", str(ts),
        "--avf-policy", "bnn",
        "--training-progress-filter", "50",
        "--oversample", str(oversample),
        "--n-epochs", "50",
        "--learning-rate", str(lr),
        "--batch-size", str(bs),
        "--patience", "10",
        "--hidden-layer-size", str(hidden),
        "--layers", str(layers),
        "--seed", str(seed),
        "--heldout-test-file", "heldout-set-seed-21-0.2-split-5-filter-cls.npz",
        "--model", str(model)
    ]

    # Only add if True
    if under:  
        cmd.append("--under")
    if augment:  
        cmd.append("--augment")
    if wl:  
        cmd.append("--weight-loss")

    print(f"Running run {run_idx} / {total_runs}")
    print(f"Running: model={model}, layers={layers}, lr={lr}, under={under}, oversample={oversample}, seed={seed}, hidden={hidden}, test_split={ts}, batch_size={bs}, augment={augment}, weight_loss={wl}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Capture output in real-time from the printed results (avf/train.py)
    final_results = None
    for line in process.stdout:
        #print(line.strip())
        if "FINAL_RESULTS:" in line:
            json_part = line.strip().split("FINAL_RESULTS:")[1]
            final_results = json.loads(json_part)

    process.wait()

    # Save to CSV
    if final_results:
        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                model, layers, lr, under, oversample, seed, hidden, ts, bs, augment, wl,
                final_results.get('accuracy'),
                final_results.get('precision'),
                final_results.get('recall'),
                final_results.get('f_measure'),
                final_results.get('auc_roc'),
                final_results.get('test_loss'),
                final_results.get('best_epochs')
            ])
    else:
        print("WARNING: No FINAL_RESULTS found for this run.")

print("Grid search completed. Results saved to", output_file)
print("Time elapsed: {}s".format(time.time() - start_time))