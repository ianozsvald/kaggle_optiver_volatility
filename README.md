
# Notes

## First attempt

### 2021-07-23

Going forwards:
 * need to include the WAP calculation as a baseline (for autocorrelation on bids/asks), plus my 2 features
 * can add more features on simple RF
 * want to add yellowbrick for prediction display
 * want to add 3fold cross val

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

#### tutorial notebook

* https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data?scriptVersionId=67183666#Competition-data
* WAP is important

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