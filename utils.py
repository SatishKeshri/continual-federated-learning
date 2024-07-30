from typing import Any, Dict, List
import argparse
import os
import copy
import torch



def average_weights(weights: List[Dict[str, torch.Tensor]], state_dict=True) -> Dict[str, torch.Tensor]:
    if state_dict == False:
        weights = [model.state_dict() for model in weights]
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--dataset", type=str, default="PermutedMNIST") #["MNIST", "PermutedMNIST"]
    parser.add_argument("--model_name", type=str, default="MLP")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--alpha", type=float, default=0.1) #Non-iid: 0.1, IID: 100000.0
    parser.add_argument("--n_clients", type=int, default=5)
    parser.add_argument("--frac", type=float, default=1)

    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--n_client_epochs", type=int, default=3) #[3,5]
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--initial_buffer_size", type=int, default=100)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_retrain", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--momentum_retrain", type=float, default=0.9)
    parser.add_argument("--memory_scheme", type=str, default="class_balanced")

    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()
