import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if os.environ.get("USER") == "ian":
    ROOT = "/home/ian/data/kaggle/optiver_volatility/"
else:
    ROOT = "/kaggle/input/optiver-realized-volatility-prediction"
print(f"Utility says ROOT is {ROOT}")

TRAIN_CSV = os.path.join(ROOT, "train.csv")
TEST_CSV = os.path.join(ROOT, "test.csv")


def get_training_stock_ids(parquet_folder="book_train.parquet"):
    return [
        int(s.split("=")[1]) for s in os.listdir(os.path.join(ROOT, parquet_folder))
    ]

def make_unique_time_ids(all_time_ids, test_size=0.33):
    first_n_upper_bound = int(len(all_time_ids) * (1 - test_size))
    test_size = len(all_time_ids) - first_n_upper_bound
    print(f"Taking {first_n_upper_bound:,} for train and {test_size:,} for test")
    #return all_time_ids[:first_n_upper_bound], all_time_ids[first_n_upper_bound:]
    return set(all_time_ids[:first_n_upper_bound]), set(
        all_time_ids[first_n_upper_bound:]
    )


def rmspe_score(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
