import argparse


def get_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--wandb", default=False, action=argparse.BooleanOptionalAction,
    help="True or False"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000,
    help="Number of training epochs (int, default: 10,000)"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=2_048,
    help="Batch size (int, default: 2,048)"
  )

  parser.add_argument(
    "-r", "--learning_rate", type=float, default=1e-3,
    help="Learning rate (float, default: 1e-3)"
  )

  parser.add_argument(
    "-v", "--validation_intervals", type=int, default=10,
    help="Number of training epochs between validations (int, default: 10)"
  )

  return parser
