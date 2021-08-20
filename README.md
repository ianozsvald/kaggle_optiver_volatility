
# Notes

## First attempt

### 2021-08-19

Working on `20210819_light` notebook, locally. 

* ~~Generalise wap and make_realized_volatility fns~~
* ~~Add linear and uniform wap2~~
* Profile the wap calculations - what's slow?
* Figure out what drives the wap calculation
* GroupKFold on small groups, check it is reproducible
* Try "from itables import init_notebook_mode" again?

During an investigation did try to change the weights to include
* uniform (all 1s)
* linear (0.1 to 1.0)
* geometric (exponential increase)
* half0half1 (0s...1s...)

_insight_ The outcome appeared to be that uniform and linear were useful, half0half1 reasonably useful, geometric somewhat harmful

_insight_ adding Numba to the weighted calc fn is faster, but we have to clean up the implementation and remove strings

```
20210819_light
Outcome:
* xgboost 30s and lgbm 2s for similar scores

Note that the data load for all parquet takes 400s on laptop (8 workers)

Top features:
0.7113 	log_return1_linear
0.1603 	log_return1_uniform
0.0770 	log_return2_linear
0.0142 	log_return2_uniform
0.0057 	size
0.0047 	stock_id
0.0036 	log_return2_half0half1
0.0031 	log_return1_half0half1 

xgboost
In [67] used 0.0000 MiB RAM in 35.65s, peaked 0.00 MiB above current, total RAM usage 963.21 MiB
r^2 score 0.818 on 107,286 predictions

lgbm
In [56] used 53.2539 MiB RAM in 1.48s, peaked 107.98 MiB above current, total RAM usage 757.58 MiB
r^2 score 0.819 on 107,286 predictions

gbmreg
In [69] used -0.0273 MiB RAM in 299.54s, peaked 0.03 MiB above current, total RAM usage 963.34 MiB
r^2 score 0.813 on 107,286 predictions

rfreg
In [71] used 2568.6680 MiB RAM in 223.72s, peaked 0.00 MiB above current, total RAM usage 3532.01 MiB
r^2 score 0.814 on 107,286 predictions

HistGradientBoostingRegressor
In [90] used 21.0547 MiB RAM in 2.41s, peaked 118.21 MiB above current, total RAM usage 3577.47 MiB
r^2 score 0.809 on 107,286 predictions # worst score?
```

```
20210819_light_kaggle
lightgbm, all features
```

### diagnostics first crack

`20210802_light_diagnostics` used

```
import shap
explainer = shap.Explainer(est)
shap_values = explainer(X_test[:5])

# visualize the first prediction's explanation
from shap.plots import _waterfall
# https://github.com/slundberg/shap/issues/2140
_waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0].values, X_test.iloc[0] )

shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values) # feature importances equivalent plot

### 2021-08-02

Working on `20210802_light` which can run on Kaggle and locally, it can use 0% or e.g. 25% of data as a test set.

```
RFReg 10 est (20s)
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'log_return', 'stock_id']
r^2 score 0.770 on 107,293 predictions
# note log_return is significantly the most predictive feature, then bid_size1_var/mean and ask_size1_mean/var as much less important, then stock_id

RFReg 50 est
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'log_return', 'stock_id']
r^2 score 0.788 on 107,293 predictions

REReg 100 est (2.5mins)
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'log_return', 'stock_id']
RandomForestRegressor(n_jobs=-1)
r^2 score 0.791 on 107,293 predictions

GBReg 100 est (3.5 mins, single core)
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'log_return', 'stock_id']
GradientBoostingRegressor()
r^2 score 0.795 on 107,293 predictions
and log_return is almost everything!

Kaggle submission scores 0.3134 (up from 0.45306), rank 1220/1559 (78th percentile) via https://www.kaggle.com/ianozsvald/20210802-light/

Adding size and also wap 3 items for mean/std
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_size2_var', 'ask_size2_var', 'wap_var', 'wap_numerator_var', 'wap_denominator_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'bid_size2_mean', 'ask_size2_mean', 'wap_mean', 'wap_numerator_mean', 'wap_denominator_mean', 'size', 'log_return', 'stock_id']
GradientBoostingRegressor()
r^2 score 0.799 on 107,289 predictions (note this will be a different train/test split than before)
BUT adding these hurt the public score (0.315!)
```

Each of the above still underpredict, with occasional large outliers.

Observation - of the basic column statistics the mean/var of order size is consistently more important than the mean/var of price - do we even need price? However removing the 4 prices seems to hurt predictive ability on 2 random runs (4 runs total - 2 with and 2 without), so I'm keeping them.

Note that a re-run of the notebook is entirely deterministic now. Using numpy's `default_rng` solved that.

Using RFReg and a linear weighted realized volatility in addition to regular volatility I get
```
USE_ALL_STOCK_IDS: True
112 unique stock ids, test set is 25.0%
Features: ['bid_price1_var', 'ask_price1_var', 'bid_price2_var', 'ask_price2_var', 'bid_size1_var', 'ask_size1_var', 'bid_size2_var', 'ask_size2_var', 'bid_price1_mean', 'ask_price1_mean', 'bid_price2_mean', 'ask_price2_mean', 'bid_size1_mean', 'ask_size1_mean', 'bid_size2_mean', 'ask_size2_mean', 'size', 'log_return1', 'log_return2', 'log_return1_linear_weight', 'stock_id']
RandomForestRegressor(n_jobs=-1, random_state=2)
r^2 score 0.811 on 107,286 predictions
top features
log_return1_linear_weight, log_return1, log_return2, size and then stats on bid/ask sizes
Kaggle 0.29276 rank 1179/1643 (71st percentile)

Top features were:
0.716272 	log_return1_linear_weight
0.116127 	log_return1
0.028613 	log_return2
0.018641 	size
0.011928 	bid_size1_var
0.010146 	bid_size2_var

The above locally on all data takes circa 200s to train with RFReg
```

TODO

* Find biggest/smallest targets, check time series for these
* ~~Count nbr of time segments (circa 3.8k)~~~, check how often we have all 600 seconds
* Make notes on how I might sample future from log return distributions
* Dig into the outliers - find the worst cases and view their timeseries
* want to add 3fold cross val
* Use https://github.com/mwouts/itables on single-stock investigator notebook
* ~~try var of wap, wap_denom etc~~ (possibly these didn't help, not clear)
* try time-weighted wap so most of impact is in 500-599 seconds
* https://pypi.org/project/sklearn-gbmi/ consider for interactions on GBReg
* could try some light hyperparam sweeps e.g. nbr estimators or max depth, could try xgboost or lightgbm

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
 * ~~need to include the WAP calculation as a baseline (for autocorrelation on bids/asks), plus my 2 features~~
 * ~~can add more features on simple RF~~
 * ~~want to add yellowbrick for prediction display~~
 * ~~shouldn't use 75% for training when submitting on kaggle~~
 * ~~nice speedup to try? https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/255473~~
 
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
pip install itables # https://github.com/mwouts/itables
conda install flake8 pandas-vet shap xgboost lightgbm eli5
```
