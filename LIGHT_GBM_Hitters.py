################################
# HITTERS
################################
#Veri Seti Değişkenleri

"""
•AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş
sayısı

•Hits: 1986-1987 sezonundaki isabet sayısı

•HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı

•Runs: 1986-1987 sezonunda takımına kazandırdığı sayı

•RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı

•Walks: Karşı oyuncuya yaptırılan hata sayısı

•Years: Oyuncunun major liginde oynama süresi (sene)

•CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı

•CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı

•CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı

•CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı

•CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı

•CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı

•League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N
seviyelerine sahip bir faktör

•Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve
W seviyelerine sahip bir faktör

•PutOuts: Oyun icinde takım arkadaşınla yardımlaşma

•Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı

•Errors: 1986-1987 sezonundaki oyuncunun hata sayısı

•Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)

•NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve
N seviyelerine sahip bir faktör

"""

import numpy as np
import pandas as pd
import missingno as msno
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from helpers.data_prep import *
from helpers.eda import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


df = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_07\hitters.csv')
df.head()

check_df(df)

# Eksik değer kontrolü
df.isnull().sum()

#Salary bağımlı değişkeninde 59 adet NA değer var.

df.dropna(inplace=True)
df.isnull().sum()

df.shape
#(263, 20)


######################
# OUTLIERS
######################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

#Observations: 263
#Variables: 20
#cat_cols: 3
#num_cols: 17
#cat_but_car: 0
#num_but_cat: 0

for col in num_cols:
    col, outlier_thresholds(df, col)

for col in num_cols:
    col, check_outlier(df, col)

for col in num_cols:
    replace_with_thresholds(df, col)

df.head()


######################
# KORELASYON ANALİZİ
######################
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f", cmap = "viridis", figsize=(11,11))
plt.title("Değişkenler Arasındaki Korelasyon")
plt.show()


###########################
# 1. FEATURE ENGINEERING
###########################

df['AtBatRatio'] = df['AtBat'] / df['CAtBat']
df['HitsRatio'] = df['Hits'] / df['CHits']
df['HmRunRatio'] = df['HmRun'] / df['CHmRun']
df['Runs_ratio'] = df['Runs'] / df['CRuns']
df['RBI_ratio'] = df['RBI'] / df['CRBI']
df['Walks_ratio'] = df['Walks'] / df['CWalks']

df['Avg_AtBat'] = df['CAtBat'] / df['Years']
df['Avg_Hits'] = df['CHits'] / df['Years']
df['Avg_Runs'] = df['CRuns'] / df['Years']
df['Avg_RBI'] = df['CRBI'] / df['Years']
df['Avg_Walks'] = df['CWalks'] / df['Years']

df.loc[(df['Years'] <= 0), 'EXPERIENCE'] = 'Noob'
df.loc[(df['Years'] > 7) & (df['Years'] <= 14), 'EXPERIENCE'] = 'Experienced'
df.loc[(df['Years'] > 14) & (df['Years'] <= 40), 'EXPERIENCE'] = 'Highly Experienced'
df.loc[(df['Years'] > 40), 'EXPERIENCE'] = 'Senior'
df["Years"] = df["Years"].astype("O")


######################
# LABEL ENCODING
######################

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

########################
# ONE HOT ENCODING
########################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, ohe_cols)


df.head()

#############################################
# STANDART SCALER
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

df.columns = [col.upper() for col in df.columns]

#######################################
# LightGBM: Model & Tahmin
#######################################

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgb_params = {"learning_rate": [0.01, 0.1, 0.3, 0.5],
               "n_estimators": [500, 1000, 2000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgb_cv_model = GridSearchCV(lgb_model,
                             lgb_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgb_cv_model.best_params_

#######################################
# Final Model
#######################################

lgb_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 0.6132442643729871

y_pred = lgb_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.4280364679506186


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgb_tuned, X_train)