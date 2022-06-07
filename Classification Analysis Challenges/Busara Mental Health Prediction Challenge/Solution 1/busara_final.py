
# coding: utf-8

# ## Winning code (simplified): Busara Mental Health Challenge by Steven Simba

# In[1]:


import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train.head()


# ## Data Cleaning

# In[2]:


seps = df_train.shape[0]
comb = pd.concat([df_train, df_test], axis=0)
comb['age'] = comb['age'].apply(lambda x: str(x) )
comb['age'] = comb['age'].apply(lambda x: str(0) if x == ".d" else x)
comb['age'] = comb['age'].apply(lambda x: float(x))

le = LabelEncoder()

comb['survey_date'] = comb['survey_date'].apply(lambda x: str(x))
le.fit(comb['survey_date'])
comb['survey_date'] = le.transform(comb['survey_date'])

colNull = comb.isnull().sum()
colNull = [keys for keys, values in colNull.items() if values > 0]
for i in colNull:
    comb[i] = comb[i].interpolate()


# In[3]:


train = comb[:seps]
test = comb[seps:]
train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)

y_train = train['depressed']
x_train = train.drop(labels=['depressed'], axis=1)
x_test = test.drop(labels=['depressed'], axis=1)


# ## Top score was a product of blending 3 models; random forest, gradient boosting and extreme gradient boosting

# In[4]:


model = GradientBoostingClassifier(n_estimators=90, max_depth=3, random_state=8) 
model.fit(x_train,y_train)
gb_pred = model.predict(x_test)

model= xgb.XGBClassifier(seed=3)
model.fit(x_train, y_train)
p_pred = model.predict_proba(x_test)

xgb_pred = []
for pp in p_pred:
    if 0.5 < pp[1] < 0.6:
        xgb_pred.append(1)
    else:
        xgb_pred.append(0)

        
model = RandomForestClassifier(random_state=3, n_estimators=20)
model.fit(x_train, y_train)
rf_pred = model.predict(x_test)

blend = []
for p in range(len(gb_pred)):
    if (gb_pred[p] > 0) | (xgb_pred[p] > 0) | (rf_pred[p] > 0):
        blend.append(1)
    else:
        blend.append(0)

submiss = pd.DataFrame({"surveyid": x_test['surveyid'],  "depressed": blend})
submiss = submiss[['surveyid', 'depressed']]
submiss.to_csv("gfinal.csv", index = False)

