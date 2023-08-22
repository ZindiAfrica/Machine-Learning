import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT/"data"
SUBMISSIONS = ROOT / "submissions"


def read_train():
    train_file = DATA / "Train.csv"
    return pd.read_csv(train_file)


def read_train_game_stats():
    train_game_stats_file = DATA / "train_game_statistics.csv"
    return pd.read_csv(train_game_stats_file)


def read_test_game_stats():
    test_game_stats_file = DATA / "test_game_statistics.csv"
    return pd.read_csv(test_game_stats_file)


def read_test():
    test_file = DATA / "Test.csv"
    return pd.read_csv(test_file)


def read_sample_submission():
    ss_file = DATA / "SampleSubmission.csv"
    return pd.read_csv(ss_file)
