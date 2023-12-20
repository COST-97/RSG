import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--name", type=str, default="default")

args, _ = parser.parse_known_args()
