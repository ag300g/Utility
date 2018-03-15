df['vlt_mean'].fillna(14, inplace=True)
df['vlt_std'].fillna(10 ** -11, inplace=True)
df['vlt_std'].replace(0, 10 ** -11, inplace=True)
# %%
# compute VLT quarterly distribution (assume normal)
for i in range(1, 10):
    p = i * 0.1
    df['quantile%.2f' % p] = norm.ppf(p, df['vlt_mean'].values, df['vlt_std'].values)  ## 增加了9列, 分位数列, 并没有分训练集和测试集

# %%
# feature type convesion string to category
df['settle_mode_cd'] = df['settle_mode_cd'].astype('category')
df['delv_way_cd'] = df['delv_way_cd'].astype('category')
# %%
# define features
feature_columns_to_use = ['uprc', 'pur_type_cd', 'account_period', 'pur_bill_attribute_cd', 'settle_mode_cd',
                          'delv_way_cd', 'sale_plat_cd',
                          'store_id', 'int_org_num', 'contract_stk_prc', 'item_first_cate_cd', 'item_second_cate_cd',
                          'item_third_cate_cd', 'brand_code', 'support_cash_on_deliver_flag', 'vlt_mean', 'vlt_std',
                          'vlt_count',
                          'vlt_min', 'vlt_max', 'wt', 'vlt_mean_season', 'width', 'height', 'calc_volume', 'len',
                          'vlt_variance_season',
                          'qtty_sum', 'qtty_min', 'qtty_max', 'qtty_mean', 'qtty_std', 'amount_sum', 'amount_min',
                          'amount_max',
                          'amount_mean', 'amount_std', 'vendor_vlt_count', 'vendor_vlt_mean', 'vendor_vlt_std',
                          'vendor_qtty_sum',
                          'vendor_qtty_min', 'vendor_qtty_max', 'vendor_qtty_mean', 'vendor_qtty_std',
                          'vendor_amount_sum',
                          'vendor_amount_min', 'vendor_amount_max', 'vendor_amount_mean']
feature_to_use = tuple(feature_columns_to_use)
nonnumeric_columns = ['pur_type_cd', 'pur_bill_attribute_cd', 'delv_way_cd', 'store_id', 'item_first_cate_cd',
                      'settle_mode_cd',
                      'item_second_cate_cd', 'item_third_cate_cd', 'int_org_num', 'brand_code',
                      'support_cash_on_deliver_flag', 'sale_plat_cd']
# %%
# define training and test sets
df_train_all = df.loc[df['complete_dt'] <= '2017-09-20']
df_test_all = df.loc[df['complete_dt'] > '2017-09-20']
print(df_train_all.shape)
print(df_test_all.shape)
# %%
# train the model
df_train = df_train_all[list(feature_to_use)]
train_y = df_train_all['VLT']
gbm = {}
features = {}
for i in range(1, 10):
    p = i * 0.1
    b = "%.2f" % p
    features[b] = []
    train_X = df_train
    train_X['quantile%.2f' % p] = df_train_all['quantile%.2f' % p]
    features[b] = list(feature_to_use)
    features[b].append('quantile%.2f' % p)  ## 分别把某一个 quantile 作为特征加入模型
    train_X = train_X[features[b]]
    gbm[b] = lgb.LGBMRegressor(alpha=p, objective='quantile', num_leaves=25, learning_rate=0.3, n_estimators=200)
    gbm[b].fit(train_X, train_y, eval_set=[(train_X, train_y)], eval_metric='l1', early_stopping_rounds=5,
               verbose=False, categorical_feature=nonnumeric_columns)
    print(gbm[b])
# %%
# save gbm model into a file named model.p
pickle.dump(gbm, open("model.p", "wb"))
# %%
# load gbm model
gbm = pickle.load(open("model.p", "rb"))
# %%

# compute VLT quarterly historical distribution
dict_quantile = {}
for i in range(1, 10):
    p = i * 0.1
    dict_quantile['%.2f' % p] = df_test_all['quantile%.2f' % p] ## 取了test集中的所有行

# %%

"""
测试数据
dict_quantile = {}
for i in range(1, 10):
    p = i * 0.1
    dict_quantile['%.2f' % p] = list(np.arange(1+i,10+i))

"""

# put the results into list VLT_test_distribution
distribution = list(dict_quantile.values())
VLT_test_distribution = np.vstack(distribution).T.tolist()  ## 按列摆放, 每一列是一个分位点
# %%
# convert vlt_distribution_data to baseline ndarray
y_base = {}
for i in range(1, 10):
    p = i * 0.1
    y_base['%.2f' % p] = []

for index, num in enumerate(VLT_test_distribution):
    for i in range(1, 10):
        p = i * 0.1
        y_base['%.2f' % p].append(num[i - 1])


# %%
# define quantile loss
def quantile_loss(quant_alpha, y, y_pred):
    loss = np.nanmean((quant_alpha - 1.0) * (y - y_pred) * (y < y_pred) + quant_alpha * (y - y_pred) * (y >= y_pred))
    return loss


# %%
# evaluate quantile loss
y_pred = {}
df_test = df_test_all[list(feature_to_use)]
test_y = df_test_all['VLT']
features = {}
for i in range(1, 10):
    p = i * 0.1
    features['%.2f' % p] = []
    test_X = df_test
    test_X['quantile%.2f' % p] = df_test_all['quantile%.2f' % p]
    features['%.2f' % p] = list(feature_to_use)
    features['%.2f' % p].append('quantile%.2f' % p)
    test_X = test_X[features['%.2f' % p]]
    y_pred['%.2f' % p] = gbm['%.2f' % p].predict(test_X)
    print('quantile: %.2f' % p)
    print('    gbm_loss:  %.3f' % (quantile_loss(p, test_y, y_pred['%.2f' % p])))
    print('    base_loss: %.3f' % (quantile_loss(p, test_y, y_base['%.2f' % p])))

# %%
# feature importance (0.9)
dict_feature_importances = dict(zip(features['0.90'], list(gbm['0.90'].feature_importances_)))
sorted_dict_feature_importances = sorted(dict_feature_importances.items(), key=operator.itemgetter(1), reverse=True)
print('Feature importances:', sorted_dict_feature_importances)

