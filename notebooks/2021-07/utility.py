import os
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


def get_data(verbose=True, stock_ids=None):
    """stock_ids can be None (uses 0..9 by default) or 'all' for all 
    stocks or a specified list of numeric ids"""
    items_in_folder = os.listdir(os.path.join(ROOT, "book_train.parquet"))
    if verbose:
        print(
            f"""There are {len(items_in_folder)} items in the folder"""
            """and they look like {items_in_folder[:5]}"""
        )

    # if stock_ids == 'all':
    #    1/0
    #    #stock_ids = range(127) # not a contiguous range!
    #    #stock_ids = get_training_stock_ids()
    # if not stock_ids:
    #    stock_ids = range(10) # [0..126] files in the total set
    if True:
        # memory efficient but needs to load everything!
        stock_filenames = []
        #for stock_id in tqdm(stock_ids):
        #    assert isinstance(stock_id, int)
        #    stock_filenames.append(os.path.join(ROOT, f"book_train.parquet/stock_id={stock_id}"))
        df_book_train = pd.read_parquet(os.path.join(ROOT, "book_train.parquet"))

    if False:
        df_book_trains = []
        for stock_id in tqdm(stock_ids):
            assert isinstance(stock_id, int)
            df_book_train_stock_X = pd.read_parquet(
                os.path.join(ROOT, f"book_train.parquet/stock_id={stock_id}")
            )
            df_book_train_stock_X["stock_id"] = stock_id
            df_book_trains.append(df_book_train_stock_X)
        #df_book_train = pd.concat(df_book_trains)
        df_book_train = pd.concat(df_book_trains, copy=False)

    if False:
        df_book_train = None
        for stock_id in tqdm(stock_ids):
            assert isinstance(stock_id, int)
            df_book_train_stock_X = pd.read_parquet(
                os.path.join(ROOT, f"book_train.parquet/stock_id={stock_id}")
            )
            df_book_train_stock_X["stock_id"] = stock_id
            if df_book_train is None:
                df_book_train = df_book_train_stock_X
            else:
                df_book_train = pd.concat((df_book_train, df_book_train_stock_X),)
                
                
    if False:
        # doesn't even finish...
        from dask.distributed import Client
        if 'client' not in dir():
            # useful for Pandas - no threads (Pandas not GIL-friendly), many processes
            # and enough memory to not max out my laptop
            client = Client(processes=True, n_workers=2, 
                            threads_per_worker=1, memory_limit='10GB')
        print(client) # show client details                

        import dask.dataframe as dd
        ddf = dd.read_parquet(path=os.path.join(ROOT, 'book_train.parquet'))
        df_book_train = ddf.compute()    

    if verbose:
        print(
            f"Loaded {df_book_train.shape[0]:,} rows for book_train on {len(stock_ids)} stock_ids"
        )

    df_train_all = pd.read_csv(TRAIN_CSV)
    training_rows_was = df_train_all.shape[0]
    df_train_all = df_train_all.query("stock_id in @stock_ids")
    training_rows_is = df_train_all.shape[0]
    if training_rows_was != training_rows_is:
        print(f"**** Had {training_rows_was:,} rows, now we have {training_rows_is:,}")
    else:
        print("Kept all training rows during get_data")

    return df_train_all, df_book_train


def make_unique_time_ids(all_time_ids, test_size=0.33):
    first_n_upper_bound = int(len(all_time_ids) * (1 - test_size))
    print(f"Taking {first_n_upper_bound} for train")
    return set(all_time_ids[:first_n_upper_bound]), set(
        all_time_ids[first_n_upper_bound:]
    )