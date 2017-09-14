# -*- coding: utf-8 -*-
"""
--------------------------------------
detect outliers and replace them
--------------------------------------
Arthur: Michael Liu
Date: 2017-09-15
LastUpdate: 2017-09-15
inputs:
-- Y: pd.Series
-- method = string
            'w': winsorize
            'm': mean+sd
            'l': lowess
-- params: list
            if winsorize: params = [persent1，percent2]
            if mean+sd: param = [how many sd]
            if lowess: param = [how many sd(n)=3, frac=0.3, it=2]
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy.stats
import copy

winsorize = scipy.stats.mstats.winsorize
lowess = sm.nonparametric.lowess
p_default = {"w":[0.00, 0.05], "m":[3], "l":[2, 0.3, 3]}  ## if user do not specify a set of params then use default

def Outlier(Y, method, params=p_default):
    if method == "l":
        if type(params) == dict:
            theta = params["l"][0]
            frac = params["l"][1]
            it = params["l"][2]
        else:
            theta = params[0]
            frac = params[1]
            it = params[2]

        lowess_Y = lowess(Y, Y.index, frac=frac, it=it)
        df = pd.DataFrame(lowess_Y)
        df.columns = ["date", "sales_smooth0"]
        df = df[["sales_smooth0"]]
        df['sales'] = Y
        df["err"] = df["sales"].astype(float) - df["sales_smooth0"]
        df["avg"] = df["err"].mean(axis=0)
        df["std"] = df["err"].std(axis=0)
        df["abnormal"] = (abs(df["err"])) >= (theta * df["std"])
        df["sales_smooth"] = df.apply(lambda df: (
        np.ceil(df["sales_smooth0"]) if np.ceil(df["sales_smooth0"]) <= np.ceil(df["sales"]) else np.ceil(
            df["sales"])) if df["abnormal"] == True else np.ceil(df["sales"]), axis=1)
        return df["sales_smooth"]

    elif method == "w":
        if type(params) == dict:
            alpha = params["w"][0]
        else:
            alpha = params[0]
        Y_w = copy.deepcopy(Y)
        Y_w.loc[:] = scipy.stats.mstats.winsorize(Y, limits=alpha)
        Y_w.rename("sales_smooth",inplace=True)
        return Y_w

    elif method == "m":
        if type(params) == dict:
            alpha= params["m"][0]
        else:
            alpha = params[0]
        Y_m = copy.deepcopy(Y)
        Y_m = pd.DataFrame(Y_m)
        Y_m.columns = ["sales"]
        Y_m.sort_values("sales", ascending=False, inplace=True)
        Y_m.reset_index(inplace=True)
        Y_m.rename(index=str, columns ={"index": "date", "sales": "sales_smooth"}, inplace=True)

        i = 0
        n = Y_m.shape[0]
        flag = 1
        while flag > 0:
            avg = Y_m.sales_smooth[(i + 1):n].mean()  ## exclude the value no less than itself
            sigma = Y_m.sales_smooth[(i + 1):n].std()
            if (Y_m.sales_smooth[i] > avg + alpha * sigma):
                ## print(Y_m.sales_smooth[i])
                i += 1
            else:
                flag = 0
        if i > 0:  ## 至少找出一个点
            Y_m.sales_smooth[:i] = np.nan
            Y_m = Y_m.sort_values('date')
            Y_m.sales_smooth = Y_m.sales_smooth.interpolate()  ## replace nan with linear interpolate
        else:
            Y_m = Y_m.sort_values('date')
        Y_m.reset_index(inplace = True)
        return Y_m["sales_smooth"]

    else:
        return Y
