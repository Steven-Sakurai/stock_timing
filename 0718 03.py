import time
from timing_module import *
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics as metrics

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

matplotlib.rcParams['figure.figsize'] = [12, 8]
plt.style.use('ggplot')

t0 = time.time()

# load OHLC data
close_df = pd.read_csv('../../data/factors_csv/close.csv', index_col=0, parse_dates=True)
high_df = pd.read_csv('../../data/factors_csv/high.csv', index_col=0, parse_dates=True)
low_df = pd.read_csv('../../data/factors_csv/low.csv', index_col=0, parse_dates=True)
open_df = pd.read_csv('../../data/factors_csv/open.csv', index_col=0, parse_dates=True)
adj_df = pd.read_csv('../../data/factors_csv/adjfactor.csv', index_col=0, parse_dates=True)

adj_last = adj_df.iloc[-1, :]
close_df = close_df * adj_df / adj_last
high_df = high_df * adj_df / adj_last
low_df = low_df * adj_df / adj_last
open_df = open_df * adj_df / adj_last

turn_df = pd.read_csv('../../data/factors_csv/turn.csv', index_col=0, parse_dates=True)
amt_df = pd.read_csv('../../data/factors_csv/amt.csv', index_col=0, parse_dates=True)
div_df = pd.read_csv('../../data/factors_csv/dividendyield2.csv', index_col=0, parse_dates=True)
mf_amt_df = pd.read_csv('../../data/factors_csv/mf_amt.csv', index_col=0, parse_dates=True)
mf_amt_close_df = pd.read_csv('../../data/factors_csv/mf_amt_close.csv', index_col=0, parse_dates=True)
mf_amt_ratio_df = pd.read_csv('../../data/factors_csv/mf_amt_ratio.csv', index_col=0, parse_dates=True)
mf_vol_ratio_df = pd.read_csv('../../data/factors_csv/mf_vol_ratio.csv', index_col=0, parse_dates=True)
mfd_buyamt_a_df = pd.read_csv('../../data/factors_csv/mfd_buyamt_a.csv', index_col=0, parse_dates=True)
mfd_sellamt_a_df = pd.read_csv('../../data/factors_csv/mfd_sellamt_a.csv', index_col=0, parse_dates=True)
dealnum_df = pd.read_csv('../../data/factors_csv/dealnum.csv', index_col=0, parse_dates=True)
pe_ttm_df = pd.read_csv('../../data/factors_csv/pe_ttm.csv', index_col=0, parse_dates=True)
volume_df = pd.read_csv('../../data/factors_csv/volume.csv', index_col=0, parse_dates=True)


##################################

## parameters
forward = 4
thresh = 0.03
h = 14
eta = 0.01
prob_ = 0.6

dis = True

stock_list = ["600000", "600016", "600019", "600028", "600029", 
"600030", "600036", "600048", "600050", "600104",
"600111", "600276", "600309", "600340", "600519",
"600547", "600585", "600606", "600690", "600703",
"600887", "600958", "600999", "601006", "601088",
"601166", "601169", "601186", "601211", "601229",
"601288", "601318", "601328", "601336", "601360",
"601390", "601398", "601601", "601628", "601668",
"601688", "601766", "601800", "601818", "601857",
"601878", "601881", "601988", "601989", "603993" ]
mycode = stock_list[30]

Bdate = pd.Timestamp('20110104')
Edate = pd.Timestamp('20180525')
#################################

# 中国银行 601988
# 中国联通 600050
# 万科A   2
# 中国石化 600028
# 中国建筑 601668
# 民生银行 600016

secid_lst = close_df.columns
assert(mycode in secid_lst)

# OHLC data
df = pd.DataFrame({
    'open': open_df.loc[:, mycode],
    'high': high_df.loc[:, mycode],
    'low': low_df.loc[:, mycode],
    'close': close_df.loc[:, mycode],
    'turn': turn_df.loc[:, mycode],
    'amt': amt_df.loc[:, mycode],
    'volume': volume_df.loc[:, mycode],
    'div': div_df.loc[:, mycode],
    'mf_amt': mf_amt_df.loc[:, mycode],
    'mf_amt_close': mf_amt_close_df.loc[:, mycode],
    'mf_amt_ratio': mf_amt_ratio_df.loc[:, mycode],
    'mf_vol_ratio': mf_vol_ratio_df.loc[:, mycode],
    'mfd_buyamt_a': mfd_buyamt_a_df.loc[:, mycode],
    'mfd_sellamt_a': mfd_sellamt_a_df.loc[:, mycode],
    'pe_ttm': pe_ttm_df.loc[:, mycode],
})

df['volume_2W'] = df.volume.rolling(14).sum()
df['volume_2Wstd'] = df.volume.rolling(14).std()


rf_bond = pd.read_csv('../../data/bond_1M.csv', index_col=0, parse_dates=True)
rf_bond.columns = ['rf']
rf_bond.index = rf_bond.index.to_series().apply(lambda x: pd.to_datetime(x))

df['rf'] = rf_bond['rf']
df['rf_pct_chg1D'] = df.rf.pct_change()
df['rf_pct_chg1M'] = df.rf.pct_change(30)
df['rf_ma'] = df.rf.rolling(14).mean()
df['rf_std'] = df.rf.rolling(14).std()

# add technical indicators

## max high,  min low
df['maxH'] = df.high.rolling(10).max()
df['minL'] = df.low.rolling(10).min()
df['maxH_d1'] = (df.maxH - df.close) / df.close 
df['minL_d1'] = (df.close - df.minL) / df.minL 

## ret1W
df['ret1W'] = df.close.pct_change().rolling(5).sum()
df['ret1W_lag1'] = df['ret1W'].shift(1)

## MASS
d = 21
df['mass'] = 0
ma_last = df.close
for i in range(2, d):
    ma = df.close.rolling(i).mean().ffill()
    df.mass = (ma_last > ma)*1 + df.mass
    ma_last = ma.copy()
    if i == d-1:
        df.mass = df.mass/i
df['mass_d1'] = df.mass.rolling(20).mean()
df['mass_d2'] = df.mass.rolling(20).std()
df['mass_d3'] = df.mass.diff(20)

## KDJ
df['slowk'], df['slowd'] = ta.STOCH(
    df.high.values,
    df.low.values,
    df.close.values,
    fastk_period=9,
    slowk_period=3, slowk_matype=0,
    slowd_period=3, slowd_matype=0)
df['J'] = 3*df.slowd - 2*df.slowk
df['J_d1'] = 1*(df.J > 100)
df['J_d2'] = 1*(df.J < 20)
df['KD_d1'] = 1*(df.slowk < df.slowd)

## MA
df['ma1'] = df.close.rolling(5).mean()
df['ma2'] = df.close.rolling(15).mean()
df['dma'] = df.ma1 - df.ma2
df['macd'] = df.dma - df.dma.rolling(14).mean()

## BBands
df['BOLL_upper'], df['BOLL_middle'], df['BOLL_lower'] = ta.BBANDS(
    df.close.values, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)
df['bb_d1'] = 1*(df.close > df.BOLL_upper)
df['bb_d2'] = 1*(df.close < df.BOLL_lower)

## Market force
df['ms'] = df.ma1.pct_change() + np.abs(df.ma1 - df.ma2)/df.ma2
df['ms_d1'] = (df.ms > 0.03)

## Price Path Efficiency
df['ppe'] = np.abs(df.close - df.close.shift(10))/np.sum(df.close.diff().rolling(10).apply(lambda x: np.sum(np.abs(x))))
df['ppe_d1'] = 1*(df.ppe > 0.3)

## RSI
df['rsi'] = ta.RSI(df.close.values, 14)

## CCI
df['cci'] = ta.CCI(df.high.values,
    df.low.values,
    df.close.values,
    14)

## CMO
df['cmo'] = ta.CMO(df.close.values, 14)

## ADX: Average Directional Movement Index, 判断盘整、振荡和单边趋势
df['adx'] = ta.ADX(df.high.values,
                   df.low.values,
                   df.close.values,
                   timeperiod=14)

## ADXR
df['adxr'] = ta.ADXR(df.high.values,
                   df.low.values,
                   df.close.values,
                   timeperiod=14)

## AROON: 价格达到近期最高值和最低值以来所经过的期间数
df['aroondown'], df['aroonup'] = ta.AROON(df.high.values,
                                          df.low.values,
                                          timeperiod=14)

## AROONOSC
df['aroonosc'] = ta.AROONOSC(df.high.values,
                             df.low.values,
                             timeperiod=14)


df['bop'] = ta.BOP(df.open.values,
                   df.high.values,
                   df.low.values,
                   df.close.values)

df['cci'] = ta.CCI(df.high.values,
                   df.low.values,
                   df.close.values,
                   timeperiod=14)


df['trix'] = ta.TRIX(df.close.values, timeperiod=30)

df['bbands1'] = 1*(df.close > df.BOLL_upper)
df['bbands2'] = 1*(df.close < df.BOLL_lower) 

df['ATR'] = ta.ATR(df.high.values, df.low.values, df.close.values, 20)

df['return_lag1'] = df.close.pct_change()

## prediction objective
df['return'] = ( (df.close.pct_change() + 1).rolling(forward).apply(np.prod) - 1).shift(-forward)

## clear nan
df = df.fillna(method="pad").fillna(method="bfill")


model = xgb.sklearn.XGBClassifier()

rolling = 200
def get_signal(context, i):
    today  = context.trading_days[i]
    yesterday = context.trading_days[i-1]
    tmp = context.df.loc[:yesterday, :]
    
    # stop loss
    if context.account.cash_account[i-1] < 0 or context.account.total_account.pct_change(3)[i-1] < -0.02 or context.current_holding_days >= 45:
        return -1, 1

    # stop profit
    if context.account.total_account.pct_change(5)[i-1] > 0.10:
        return -1, 1

    X = tmp.iloc[(-rolling-forward):-forward, :-1].values
    y = tmp.iloc[(-rolling-forward):-forward, -1].values
    y = np.sign(y)
    y[y == 0] = 1

#    kfold = TimeSeriesSplit(n_splits=cv)
#    n_estimators = list(range(5, 12, 2))
#    max_depth = list(range(3, 10, 1))
#    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
#    grid_search = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=kfold, verbose=1)
#    grid_result = grid_search.fit(X, y)
#    my_model = grid_result.best_estimator_
    my_model = xgb.sklearn.XGBClassifier()
    my_fit = my_model.fit(X, y)

    features = tmp.columns[:-1][my_fit.feature_importances_ > thresh].tolist()
    if len(features) < 2:
        return 0, 0
    # the last one is yesterday's data and the return to predict
    X_s = tmp.iloc[(-rolling-forward):-forward, :]
    X_s = X_s.loc[:, features].values
    X_p = tmp.iloc[-forward:, :]    
    X_p = X_p.loc[:, features].values
    true_sign = np.mean(tmp.iloc[-forward:, -1].values)

    m1 = svm.SVC(kernel='rbf', probability=True)
    f1 = m1.fit(X_s, y)
    pred1 = np.mean( f1.predict(X_p) )
    prob1 = np.mean( f1.predict_proba(X_p).T[0] )
    context.record.iloc[i, 0] = (1 if pred1*true_sign >= 0 else -1)

    m2 = RandomForestClassifier()
    f2 = m2.fit(X_s, y)
    pred2 = np.mean( f2.predict(X_p) )
    prob2 = np.mean( f2.predict_proba(X_p).T[0] )
    context.record.iloc[i, 1] = (1 if pred2*true_sign >= 0 else -1)

    m3 = MLPClassifier()
    f3 = m3.fit(X_s, y)
    pred3 = np.mean( f3.predict(X_p) )
    prob3 = np.mean( f3.predict_proba(X_p).T[0] )
    context.record.iloc[i, 2] = (1 if pred3*true_sign >= 0 else -1)

    m4 = xgb.sklearn.XGBClassifier()
    f4 = m4.fit(X_s, y)
    pred4 = np.mean(m4.predict(X_p))
    prob4 = np.mean( f4.predict_proba(X_p).T[0] )
    context.record.iloc[i, 3] = (1 if pred4*true_sign >= 0 else -1)

    i1 = i-h if i > h else 0
    w1 = 0.25 + eta*np.sum(context.record.iloc[i1:i, 0])
    w2 = 0.25 + eta*np.sum(context.record.iloc[i1:i, 1])
    w3 = 0.25 + eta*np.sum(context.record.iloc[i1:i, 2])
    w4 = 0.25 + eta*np.sum(context.record.iloc[i1:i, 3])


    if w1 < 0:
        w1 = 0
    if w2 < 0:
        w2 = 0
    if w3 < 0:
        w3 = 0
    if w4 < 0:
        w4 = 0
    if w1+w2+w3+w4 <= 0:
        w1 = w2 = w3 = w4 = 0.25
    pred = (w1*pred1 + w2*pred2 + w3*pred3 + w4*pred4)/(w1+w2+w3+w4)
    #context.record.iloc[i, 4] = pred
    #prob = (w1*prob1 + w2*prob2 + w3*prob3 + w4*prob4)/(w1+w2+w3+w4)
    #pred = 2*(prob > prob_) - 1 
    context.record.iloc[i, 4] = (1 if pred*true_sign >= 0 else -1)
    context.record.iloc[i, 5] = true_sign
    
    if dis:
        print("top features: ")
        print(features)
        print("1st classifier's judgement: %.2f" % (pred1, ))
        print( (pred1 * true_sign) > 0 )
        print("2nd classifier's judgement: %.2f" % (pred2, ))
        print( (pred2 * true_sign) > 0 )
        print("3rd classifier's judgement: %.2f" % (pred3, ))
        print( (pred3 * true_sign) > 0 )
        print("4th classifier's judgement: %.2f" % (pred4, ))
        print( (pred4 * true_sign) > 0 )
        print("final classifier's judgement: %.2f" %(pred, ))
        print( (pred * true_sign) > 0 )

    a = context.record.iloc[i1:i, 5]
    w = len(a[a > 0]) / ( 1 + len(a[a != 0]) )
    
    increase = w * tmp.mass_d1[-1] / tmp.ATR[-1]
    if increase > 0.8:
        increase = 0.8

    if pred > 0:
      return 1, increase
    else:
      return -1, 1

C = Context(df, begin_date_=Bdate, end_date_=Edate,
  commission_=0.0, slippage_=0.0, display_=dis)


C.record = pd.DataFrame({
    "svm": np.repeat(0, C.length),
    "rf": np.repeat(0, C.length),
    "lr": np.repeat(0, C.length),
    "xgb": np.repeat(0, C.length),
    "final": np.repeat(0, C.length),
    "true": np.repeat(0, C.length),
}, index=C.trading_days)



print("--- %s seconds ---" % (time.time() - t0))
print("Backtest start!")

backtest(C, get_signal, initial=1000000)

print("--- %s seconds ---" % (time.time() - t0))

print("-- clf performance --")
a1 = C.record.iloc[:, 0]
a2 = C.record.iloc[:, 1]
a3 = C.record.iloc[:, 2]
a4 = C.record.iloc[:, 3]
b = C.record.iloc[:, 4]
print( len(a1[a1 > 0]) / len(a1[a1 != 0]) )
print( len(a2[a2 > 0]) / len(a2[a2 != 0]) )
print( len(a3[a3 > 0]) / len(a3[a3 != 0]) )
print( len(a4[a4 > 0]) / len(a4[a4 != 0]) )
print( len(b[b > 0]) / len(b[b != 0]) )

print("-----------------------")
print("benchmark: ")
Bclose = C.df.loc[C.begin_date:C.end_date].close

print("excess return's mean, std, sharpe ratio:")
print(sharpe_ratio(Bclose))

print("最大回撤率：")
print(max_drawdown(Bclose))

print("Calmar Ratio：")
print(CalmarRatio(Bclose))

print("年化收益率：")
print(annualized_return(Bclose, len(Bclose)))

print("-------Portfolio Summary---------")
perform_analysis(C)

c = 0.4 * C.account.total_account[0] / np.max(C.account.position)

compare = pd.DataFrame({
    "position": C.account.position,
    "signal": C.account.signal,
    "portfolio": C.account.total_account,
    "close": Bclose,
    "benchmark": 1000000 * Bclose / Bclose[0]
})

compare.to_csv(mycode+".csv")

compare.plot()
plt.show()

