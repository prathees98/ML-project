# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load..

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualize the data
import seaborn as sns # visualize the data
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle 
import lightgbm as lgb

import warnings

warnings.simplefilter('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

df_train = pd.read_feather('/kaggle/input/amexfeather/train_data.ftr')
df_train.drop(['target'], axis = 1, inplace = True)
df_train

df_test = pd.read_feather('/kaggle/input/amexfeather/test_data.ftr')
df_test

target = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')
target

# EDA

print(df_train.shape)
print(df_test.shape)
print(target.shape)

df_train

df_train.info(verbose=True, null_counts=True)

df_train.select_dtypes(include = ['object', 'category']).columns

categorical_cols = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
num_cols = []
for col in df_train.columns:
    if col not in categorical_cols+['customer_ID','S_2']:
        num_cols.append(col)

num_cols = np.array(num_cols)
print("categorical cols: ", categorical_cols)
print("numerical cols: ", num_cols)

df_train[categorical_cols]

def denoise(df):
    df['D_63'] = df['D_63'].apply(lambda t: {'CR': 0, 'XZ': 1, 'XM': 2, 'CO': 3, 'CL': 4, 'XL': 5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {'O': 0, 'U': 3, 'R': 2, '': -1, '-1': 1}[t]).astype(np.int8)

    return df

def variance(df):
    df_temp = df.drop(['customer_ID','S_2'], axis=1)

    var_thres = VarianceThreshold(threshold = 0)
    var_thres.fit(df_temp)

    constant_columns = [column for column in df_temp.columns if column not in df_temp.columns[var_thres.get_support()]]

    df.drop(constant_columns, axis=1, inplace = True)
    
    return df

def null_details(threshold : np.int8 = 50, cols: np.ndarray = df_train.columns, df: pd.DataFrame = df_train) -> pd.DataFrame:
    null_cols = df[cols].isnull().sum().sort_values()
    df_null = pd.DataFrame(null_cols[null_cols > 0])
    df_null[1] = df[cols].isnull().mean()*100

    return df_null[df_null[1] > threshold]

def remove_cols(df):
    return null_details(80, df.columns).index

def select_best_indeces(group: pd.DataFrame):
    return group.isnull().sum(axis = 1).sort_values().head(1).index[0]

def select_rows(df_grouped):
    return df_grouped.apply(select_best_indeces).values

def one_hot_encoding(df, cols, is_drop = True):
    for col in cols:
        print('one hot encoding:', col)
        dummies = pd.get_dummies(pd.Series(df[col]), prefix =' oneHot_%s'%col, drop_first = True)
        df = pd.concat([df, dummies], axis = 1)
    if is_drop:
        df.drop(cols, axis = 1, inplace = True)
        
    return df

def ordinal_encoding(df, cols):
    enc = OrdinalEncoder()
    df[cols] = enc.fit_transform(df[cols])
    
    return df

# pd.set_option('display.max_rows', None)
null_details(0, num_cols)

D_cols = df_train.columns[pd.Series(df_train.columns).str.startswith('D_')]
B_cols = df_train.columns[pd.Series(df_train.columns).str.startswith('B_')]
S_cols = df_train.columns[pd.Series(df_train.columns).str.startswith('S_')]
R_cols = df_train.columns[pd.Series(df_train.columns).str.startswith('R_')]
P_cols = df_train.columns[pd.Series(df_train.columns).str.startswith('P_')]

Dict = {'Delinquency': len(D_cols), 'Spend': len(S_cols), 'Payment': len(P_cols), 'Balance': len(B_cols), 'Risk': len(R_cols),}

plt.figure(figsize=(10,5))
sns.barplot(x=list(Dict.keys()), y=list(Dict.values()));
plt.legend()

grouped_df = df_train.dropna(how='all').groupby("customer_ID")
selected_indices = select_rows(grouped_df)

df_train = df_train.loc[selected_indices, :]
df_train

df_train.describe()

null_details(0, df_train.columns, df_train)

df_train = denoise(df_train)

removeble_cols = list(remove_cols(df_train))
removeble_cols.append('S_2')

for col in df_train.iloc[:, 2:].columns:
    if col not in removeble_cols:
        if col in categorical_cols:
            df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
        else:
            df_train[col] = df_train[col].fillna(df_train[col].median())

null_details(0, df_train.columns, df_train)
df_train

# EDA on test data

nans = null_details(0, df_test.columns, df_test)
print(nans)
print(nans.shape)
grouped_df_test = df_test.dropna(how='all').groupby("customer_ID")
rows_indices = select_rows(grouped_df_test)
rows_indices
df_test = df_test.loc[rows_indices, :]
df_test

null_details(0, df_test.columns, df_test)
df_test = denoise(df_test)
for col in df_test.iloc[:, 2:].columns:
    if col not in removeble_cols:
        if col in categorical_cols:
            df_test[col] = df_test[col].fillna(df_test[col].mode()[0])
        else:
            df_test[col] = df_test[col].fillna(df_test[col].median())

null_details(0, df_test.columns, df_test)
df_test['D_68'] = pd.Categorical(df_test['D_68'], categories=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ordered=False)
df_test

# Handle chatecorical data

for col in ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68']:
    le = LabelEncoder()
    le.fit(df_train[col])
    print('*****')
    print(col)
    print(le.classes_)
    print(df_test[col].unique())
    print(df_train[col].unique())
    print('*****')
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
df_train[['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68']]
ct = ColumnTransformer([('oneHot', OneHotEncoder(drop='first'), ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_68'])], remainder = 'passthrough')

train_matrix = ct.fit_transform(df_train)
test_matrix = ct.transform(df_test)
new_features = ct.get_feature_names()
train_matrix.dtype
df_train = pd.DataFrame(train_matrix, columns = new_features)
df_train.set_index('customer_ID', drop = True, inplace = True)
df_train
df_test = pd.DataFrame(test_matrix, columns = new_features)
df_test.set_index('customer_ID', drop = True, inplace = True)
df_test

# Feature selection

df_train.loc[:, pd.Series(df_train.columns).str.startswith('oneHot').values] = df_train.loc[:, pd.Series(df_train.columns).str.startswith('oneHot').values].astype('int8')
df_train.loc[:, list(df_train.select_dtypes(include=['object']).columns)] = df_train.loc[:, list(df_train.select_dtypes(include=['object']).columns)].astype('float64')
df_train.dtypes.value_counts()

cor_matrix = df_train.corr()
col_core = set()

for i in range(len(cor_matrix.columns)):
    for j in range(i):
        if(cor_matrix.iloc[i, j] > 0.9):
            col_name = cor_matrix.columns[i]
            col_core.add(col_name)
            
col_core
plt.figure(figsize=(10,5))
sns.heatmap(cor_matrix, cmap = plt.cm.Accent_r, annot = True)
dropable_cols = np.concatenate((removeble_cols, list(col_core)))
dropable_cols

df_train.drop(dropable_cols, axis = 1, inplace = True)
df_train
df_test.drop(dropable_cols, axis = 1, inplace = True)
df_test

# Train model

target.set_index('customer_ID', drop = True, inplace = True)
merged_df = pd.concat([df_train, target], axis = 1)
merged_df
X = merged_df.drop('target', axis = 1)
y = merged_df['target']
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
df_test_scaled = scaler.transform(df_test)

X_train_scaled
ran_clf = RandomForestClassifier(n_estimators = 10, max_depth = 2, random_state=0)
ada_clf = AdaBoostClassifier()
gbm_clf = GradientBoostingClassifier()
d_train = lgb.Dataset(data = X_train_scaled, label = y_train)

params = {'objective': 'binary','n_estimators': 1200,'metric': 'binary_logloss','boosting': 'gbdt','num_leaves': 90,'reg_lambda' : 50,'colsample_bytree': 0.19,'learning_rate': 0.03,'min_child_samples': 2400,'max_bins': 511,'seed': 42,'verbose': -1}

lgb_clf = lgb.train(params, d_train, 100)

pred=[]
pred_prob = []

for model in [ran_clf,ada_clf,gbm_clf]:
    model.fit(X_train_scaled, y_train)
    pred.append(model.predict(X_test_scaled))
    pred_prob.append(model.predict_proba(X_test_scaled))
predicts = [pd.Series(x) for x in pred]
probabilities = [pd.DataFrame(x, columns = [0, 1]) for x in pred_prob]
lgb_pred = lgb_clf.predict(X_test_scaled)
lgb_pred

predicts.append(pd.Series(lgb_pred))
# probabilities.append(pd.DataFrame(lgb_prob, columns = [0, 1]))
y_pred = pd.concat(predicts, axis = 1)
y_pred
y_prob = pd.concat(probabilities, axis = 1)
y_prob
lgb_clf.predict(X_test_scaled)

# AMEX metric

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
    
       def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()
    
     def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

#Performance

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Default')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
accuracy_score(y_test, y_pred.loc[:, [3]])
amex_metric_mod(y_test.values, y_prob.iloc[:, [7]].values.flatten())

amex_metric_mod(y_test.values, lgb_pred)
roc_auc_score(y_test, y_prob.iloc[:, [5]].values.flatten())
fpr, tpr, thresholds = roc_curve(y_test, y_prob.iloc[:, [7]].values.flatten())
plt.figure(figsize=(16, 9))
plot_roc_curve(fpr, tpr)

# Submission

predict_prob = lgb_clf.predict(df_test_scaled)#[:, 1].reshape(-1,1).flatten()
predict_prob
len(predict_prob)
sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': df_test.index, 'prediction': predict_prob})
output.to_csv('/kaggle/working/my_submission_7.zip', index = False, compression = 'zip')

