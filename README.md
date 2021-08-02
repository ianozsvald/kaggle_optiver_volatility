
# Notes

## First attempt

### 2021-07-30

Building on previous to simplify the data load into one of two choices with `20210728_light`.

Note that taking Alair plots out of a notebook doesn't seem to change the resident memory usage. The file on disk with Altair is however 10x the size of the no-altair nb!

### 2021-07-28

Adding some more features to RF (means and more variance)

```
est is RFReg with 10 est and defaults (note doing same with 30 est gets r^2 of 0.67)
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'stock_id']
r^2 score 0.657
```

Submitted the above to Kaggle with 30 estimators. Now I rank 1309/1409 with a score of 0.45306. 

My prediction error plot suggests I'm under-fitting on the larger values.

Attempting a first crack at using WAP (too basic) gives:

```
30 unique stock ids, test set is 25.0%
Features: ['wap_var', 'wap_mean', 'stock_id']
r^2 score 0.510 on 5,748 predictions
```


### 2021-07-23

Going forwards:
 * need to include the WAP calculation as a baseline (for autocorrelation on bids/asks), plus my 2 features
 * ~~can add more features on simple RF~~
 * ~~want to add yellowbrick for prediction display~~
 * want to add 3fold cross val
 * could try some light hyperparam sweeps e.g. nbr estimators or max depth, could try xgboost or lightgbm
 * ~~shouldn't use 75% for training when submitting on kaggle~~
 * nice speedup to try? https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/255473
 
Submitted model
```
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'stock_id']
r^2 score 0.416
Kaggle - rank 1320/1338 score 1.2 (worse than the benchmark!)
```

Built a new stratified train/test split on time_id. Load an arbitrary number of stocks. Build an estimator e.g. linear regression or random forest regressor.

Loading all training data (train.csv & book_train, not trades) takes 14GB, peaks 2.5GB above final, takes 20s to load.

It looks like whilst variance on bid/ask price has some value, the stock_id has more - perhaps local information?

```
RFRegressor, 10 trees
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'stock_id', 'time_id']
r^2 score 0.448, 0.448

112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'stock_id']
r^2 score 0.477, 0.475
- interesting! using stock_id is a good thing, time_id is a bad thing

112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'time_id']
r^2 score 0.414, 0.402

112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var']
r^2 score 0.101
```

167M rows of book_train data, 8GB in RAM.


TODO

* calculate WAP per second, for the book, all overall
* normalise teh 0th time index on test set (noted in forum)
* is query/eval faster than masking on 167M rows w/wo numexpr?
* ~~install bottleneck numexpr~~
* ~~Add Altair plots~~
* ~~Load test data~~
* ~~predict on test data~~
* ~~submit a notebook that runs, upload utility & notebook, check os.environ~~
* ~~need to use from utility import x, y, z so i can copy easily into kaggle nb~~
* ~~make a test in utility~~

#### pd read_parquet learning

* pd.read_parquet and appends into a list, 14GB overall in 21s, peaks 2.5GB above the end
* pd.read_parquet concat into a growing df, 8.4GB overall 164s, peaks 8GB above
* trying dd.from_dataframe fails on memory
* pd.read_parquet with paths, 13GB overall in 6s.

I made a book_test_local.parquet folder with 6 files to use for local testing of the prediction process

```
    stock_ids_for_small_test_set = range(120, 127)
    dfx = df_train_all.query('stock_id in @stock_ids_for_small_test_set')
    ser_row_id = dfx.reset_index()[['stock_id', 'time_id']].apply(lambda x: f"{x[0]}-{x[1]}", axis=1)
    dfx2 = dfx.reset_index()
    dfx2['row_id'] = ser_row_id
    dfx2.to_csv('test_local.csv', index=False)
```

#### tutorial notebook

* https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data?scriptVersionId=67183666#Competition-data
* WAP is important

#### feature ideas

* https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/250357 ask1+ask2 bid1+bid2 combined information
* stock 31 https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/250503 has low liquidity

### 2021-07-19

Wrote a `LinearRegression` model using `book_train`, `train.csv` (has the labels), some stocks and all time ids using r^2 metric. Used bid/ask (1 and 2) price through variance function, only used 1 or several of the resulting variances as features. The variances came from a groupby on `time_id` via `seconds_in_bucket` (range 0 to 599, not all present) with a `.var` calculation on a particular column.

Using a single train/test split (66% test) with a RF Regressor on 10 stock ids I get an r^2 of 0.3.

Plotted using Altair.

TODO:

* ~~Answer for all stocks, how many time-ids do I have? Maybe a bar chart?~~
* I need to train on all stock ids
* possibly use stock-ids as a feature
* ~~split train/test on time_id~~
* write out a test set for submission

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
conda install pytest numexpr bottleneck
```
