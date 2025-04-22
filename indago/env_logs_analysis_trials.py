import argparse
import glob
import math
import os

import numpy as np

from indago.avf.config import AVF_TRAIN_POLICIES
from indago.avf.env_configuration import EnvConfiguration
from indago.config import DONKEY_ENV_NAME, ENV_NAMES, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.envs.donkey.donkey_env_configuration import DonkeyEnvConfiguration
from indago.envs.humanoid.humanoid_env_configuration import HumanoidEnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.stats.effect_size import cohend, vargha_delaney_unpaired
from indago.stats.power_analysis import parametric_power_analysis
from indago.stats.stat_tests import mannwhitney_test
from log import Log

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder where logs are", type=str, required=True)
parser.add_argument("--env-name", help="Env name", type=str, choices=ENV_NAMES, default=None)
parser.add_argument("--avf-train-policy", help="Avf train policy", type=str, choices=AVF_TRAIN_POLICIES, default="mlp")
parser.add_argument("--type", help="The type of analysis, input or output", type=str, default='output')
parser.add_argument("--alpha", help="Statistical significance level for statistical tests", type=float, default=0.05)
parser.add_argument("--beta", help="Power level for statistical tests", type=float, default=0.8)
parser.add_argument("--adjust", help="Adjust p-values when multiple comparisons", action="store_true", default=False)
parser.add_argument("--names", nargs="+", help="Names associated to files", required=True)

args = parser.parse_args()


def get_env_config(s: str, env_name: str) -> EnvConfiguration:
    assert "FAIL -" in s, "String {} not supported".format(s)
    str_env_config = s.replace("INFO:experiments:FAIL - Failure probability for env config ", "").split(": ")[0]
    # TODO refactor
    if env_name == PARK_ENV_NAME:
        env_config_ = ParkingEnvConfiguration().str_to_config(s=str_env_config)
    elif env_name == HUMANOID_ENV_NAME:
        env_config_ = HumanoidEnvConfiguration().str_to_config(s=str_env_config)
    elif env_name == DONKEY_ENV_NAME:
        env_config_ = DonkeyEnvConfiguration().str_to_config(s=str_env_config)
    else:
        raise NotImplementedError("Unknown env name: {}".format(env_name))
    return env_config_


def get_failure_probability(s: str) -> float:
    assert "FAIL -" in s, "String {} not supported".format(s)
    return float(
        s.replace("INFO:experiments:FAIL - Failure probability for env config ", "")
        .split(": ")[1]
        .replace("(", "")
        .split(",")[0]
    )

# Get the seed by returning the last part of the string
def get_seed(s: str) -> int:
    assert "random seed" in s, f"String {s} not supported"
    return int(s.strip().split(" ")[-1])

# Get the total predictions by returning the last part of the string
def get_total_predictions(s: str) -> int:
    assert "Number of evaluation predictions" in s, f"String {s} not supported"
    return int(s.strip().split(" ")[-1])

def number_of_failures(s: str) -> list:
    assert "INFO:experiments:Failure probabilities: " in s, f"String {s} not supported"
    return (s.strip()
                .replace("INFO:experiments:Failure probabilities: [", "")
                .split(']')[0]
                .split(', '))

def get_runtimes(s: str) -> list:
    assert "INFO:Avf:Times elapsed (s): " in s, f"String {s} not supported"
    return (s.strip()
                .replace("INFO:Avf:Times elapsed (s): [", "")
                .split(']')[0]
                .split(', '))

def strip_and_clean_string(s: str) -> str:
    return (s.strip()
            .replace("%", "")
            .split(': ')[-1]
            )

def get_diversity_info(name: str) -> list:
    filename = os.path.join(args.folder, 'diversity', f'{args.type}_diversity_0-trial', f"{args.type}-diversity-{args.avf_train_policy}-{name}.txt")
    assert os.path.exists(filename), "{} does not exist".format(filename)

    diversity = []

    with open(filename, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            
            if "NO DIR" in line:
                diversity.append([0,0,0])
            elif f"WARNING:{args.type}_diversity:Not possible to cluster" in line:
                diversity.append([0,0,0])
            elif f"INFO:{args.type}_diversity:Number of clusters" in line:
                clusters = strip_and_clean_string(line)
                # diversity_clusters.append(strip_and_clean_string(line))
            elif f"INFO:{args.type}_diversity:Coverage for method" in line:
                coverage = strip_and_clean_string(line)
                # diversity_coverage.append(strip_and_clean_string(line))
            elif f"INFO:{args.type}_diversity:Entropy:" in line and "%" in line:
                entropy = strip_and_clean_string(line)
                diversity.append([clusters, coverage, entropy])
                # diversity_entropy.append(strip_and_clean_string(line))
    return diversity

if __name__ == "__main__":

    logger = Log("env_logs_analysis_trial")
    logger.info("Args: {}".format(args))

    seeds = []
    predictions = []
    timings = []
    failures = []

    failures_names = dict()

    # Get the policy name for file names
    policy = args.names[0].split('_')[-1]

    with open(os.path.join(args.folder, f"results_{args.type}_{policy}.csv"), "w", encoding="utf-8") as out_csv, open(os.path.join(args.folder, f"failures_{args.type}_{policy}.csv"), "w", encoding="utf-8") as failures_csv:
        out_csv.write('name,run_id,seed,total_predictions,avg_runtime,total_runtime,num_failures,div_clusters,div_coverage,div_entropy\n')
        failures_csv.write('name,run_id,environment\n')

        for i in range(len(args.names)):
            name = args.names[i]
            
            diversity_info = get_diversity_info(name)
            # Offset when a file doesn't exist to remain aligned!
            diversity_offset = 0

            if name == "random" or "prioritized_replay" in name:
                filenames = sorted(glob.glob(os.path.join(args.folder, f'testing_logs_{args.avf_train_policy}', name, "testing-*{}-*-trial.txt".format(name))))
            else:
                # filenames = sorted(glob.glob(os.path.join(args.folder, "testing-{}-{}-*-trial.txt".format(args.avf_train_policy, name))))
                filenames = sorted(glob.glob(os.path.join(args.folder, f'testing_logs_{args.avf_train_policy}', name, "testing-*{}-*-trial.txt".format(name))))
            assert len(filenames) > 0, "Log files for {} not found".format(name)

            for trial_num, filename in enumerate(filenames):
                assert os.path.exists(filename), "{} does not exist".format(filename)

                if name not in failures_names:
                    failures_names[name] = dict()
                if trial_num not in failures_names[name]:
                    failures_names[name][trial_num] = []

                if len(glob.glob(os.path.join(args.folder, f'replay_logs_{args.avf_train_policy}', 'replay_test_failure', f"replay_test_failure_{args.avf_train_policy}-{name}-*-{trial_num}-trial"))) <= 0:
                    out_csv.write(f'{name},{trial_num},{seed},0,0,0,0,0,0,0\n')
                    failures_csv.write(f'{name},{trial_num},None\n')
                    diversity_offset += 1
                else:
                    with open(filename, "r+", encoding="utf-8") as f:
                        for line in f.readlines():
                            if "INFO:experiments:FAIL -" in line:
                                env_config = get_env_config(s=line, env_name=args.env_name)
                                fp = get_failure_probability(s=line)
                                failures_names[name][trial_num].append((env_config, fp))
                                failures_csv.write(f'{name};{trial_num};{env_config.get_str()}\n')
                            elif "INFO:instantiate_model:Setting random seed" in line:
                                seed = get_seed(line)
                                seeds.append(seed)
                            elif "INFO:experiments:Number of evaluation predictions" in line:
                                prediction = get_total_predictions(line)
                                predictions.append(prediction)
                            elif "INFO:Avf:Times elapsed (s)" in line:
                                timings = get_runtimes(line)
                            elif "INFO:experiments:Failure probabilities: " in line:
                                failures = number_of_failures(line)

                    total_runtime = sum([float(time) for time in timings])
                    total_failures = sum([float(x) for x in failures])
                    
                    print(trial_num)
                    # Writes results for each line
                    out_csv.write(f'{name},{trial_num},{seed},{prediction},{round(total_runtime / len(timings), 2)},{round(total_runtime, 2)},{int(total_failures)},{diversity_info[trial_num-diversity_offset][0]},{diversity_info[trial_num-diversity_offset][1]},{diversity_info[trial_num-diversity_offset][2]}\n')

    lengths = [len(failures_names[name]) for name in failures_names.keys()]
    assert len(lengths) > 0 and sum(lengths) == lengths[0] * len(
        lengths
    ), "Number of trials must be the same for all names {}: {}".format(args.names, lengths)

    num_trials = lengths[0]
    if len(failures_names) == 1:
        # no statistical comparison
        num_failures = [len(failures_names[args.names[0]][key]) for key in failures_names[args.names[0]]]
        method = args.names[0]
        print("Failures {}: {}".format(method, num_failures))
    else:
        # statistical analysis (no adjust)
        for i in range(len(failures_names)):
            num_failures_a = [len(failures_names[args.names[i]][key]) for key in failures_names[args.names[i]]]
            method_a = args.names[i]
            for j in range(i + 1, len(failures_names)):
                num_failures_b = [len(failures_names[args.names[j]][key]) for key in failures_names[args.names[j]]]
                method_b = args.names[j]
                _, p_value = mannwhitney_test(a=list(num_failures_a), b=list(num_failures_b))
                print("Failures {}: {}, Failures {}: {}".format(method_a, num_failures_a, method_b, num_failures_b))
                if p_value < args.alpha:
                    eff_size_magnitude = vargha_delaney_unpaired(a=list(num_failures_a), b=list(num_failures_b))
                    print(
                        "{} ({}) vs {} ({}), p-value: {}, effect size: {}, significant".format(
                            method_a, np.mean(num_failures_a), method_b, np.mean(num_failures_b), p_value, eff_size_magnitude
                        )
                    )
                else:
                    effect_size, _ = cohend(a=list(num_failures_a), b=list(num_failures_b))
                    if math.isclose(effect_size, 0.0):
                        print(
                            "{} ({}) vs {} ({}), not significant".format(
                                method_a, np.mean(num_failures_a), method_b, np.mean(num_failures_b)
                            )
                        )
                    else:
                        sample_size = parametric_power_analysis(effect=effect_size, alpha=args.alpha, power=args.beta)
                        if sample_size > num_trials:
                            print(
                                "{} ({}) vs {} ({}), sample size: {}".format(
                                    method_a,
                                    np.mean(num_failures_a),
                                    method_b,
                                    np.mean(num_failures_b),
                                    int(sample_size) if sample_size != math.inf else math.inf,
                                )
                            )
                        else:
                            print(
                                "{} ({}) vs {} ({}), not significant ({})".format(
                                    method_a,
                                    num_failures_a,
                                    method_b,
                                    num_failures_b,
                                    int(sample_size) if sample_size != math.inf else math.inf,
                                )
                            )
