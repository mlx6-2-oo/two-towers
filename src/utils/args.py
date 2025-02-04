import argparse

def parse_wandb_args(description="Two Towers Model"):
    """Simple argument parser that just handles the wandb flag."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    return parser.parse_args() 