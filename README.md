
# Notes

## First attempt

### 2021-07-15

Data extracted to ~/data/kaggle/optiver_volatility. `book_train` appears to contain circa 127 subfolders, 1 per stock, with 1 parquet file of circa 20MB per stock. `trade_train` is the same for 1-2MB parquet files. This might be 112 non-sequential items.

`train.csv` contains

```
$ more train.csv 
stock_id,time_id,target
0,5,0.004135767
0,11,0.001444587
```

Training has 428k items. 3.8k unique times (not necessarily unique to each stock?), 112 unique stocks. Stocks seem to have 111 or 112 time stamps, so they timestamps are almost all shared.

In `20210715 first eda` if I take `bid_train` for stock id 0 or 1, take the variance by `time_id` for the bid or ask columns (2 of each), then join to `trade_train` on `target`, we see a positive relationship for variance on price to the target and a triangular relationship on volume to target (low volume=weak positive, then for larger volume weak negative).

#### uncertainties

* don't know aim of competition (need to read)
* believe that `time_id` is non-sequential, so we can't try to make inferences
* test configuration is deliberately very light on local data
* appears to be regression (not CI)
* book_train stock_id 99 is weird (many items, single time_id items)
* do the relationships seen on 07-15 for price vs target positive relationships hold up?



# Environment

## 2021-07-15

```
conda create -n optiver_volatility python=3.8 pandas jupyterlab scikit-learn pandas-profiling matplotlib altair ipython_memory_usage
```