import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
train=pd.read_csv("./Data/train.zip")
test=pd.read_csv("./Data/test.csv")
train.drop(['payment_method', 'payment_receipt', 'seat_number',"travel_to"],axis=1,inplace=True)
test.drop(["travel_to"],axis=1,inplace=True)

number_of_ticket=train.groupby(["ride_id"]).ride_id.count().rename("number_of_ticket").reset_index()
ride_ids=[]
travel_dates=[]
travel_times=[]
tavel_froms=[]
car_types=[]
max_capacitys=[]
for ride_id in train.ride_id.unique(): 
    ride_ids.append(ride_id)
    travel_dates.append(train[train.ride_id==ride_id].travel_date.unique()[0])
    travel_times.append(train[train.ride_id==ride_id].travel_time.unique()[0])
    tavel_froms.append(train[train.ride_id==ride_id].travel_from.unique()[0])
    car_types.append(train[train.ride_id==ride_id].car_type.unique()[0])
    max_capacitys.append(train[train.ride_id==ride_id].max_capacity.unique()[0])
train=pd.DataFrame()
train["ride_id"]=ride_ids
train["travel_date"]=travel_dates
train["travel_time"]=travel_times
train["travel_from"]=tavel_froms
train["car_type"]=car_types
train["max_capacity"]=max_capacitys
train=train.merge(number_of_ticket,how="left",on="ride_id")
def fixe_time(x):
    date=x.split("-")
    date[-1]="20"+date[-1]
    return "-".join(date)
train["time"]=(train["travel_date"].apply(fixe_time)+" "+train["travel_time"]).astype(str)
test["time"]=(test["travel_date"]+" "+test["travel_time"]).astype(str)
train["time"]=pd.to_datetime(train["time"],format='%d-%m-%Y %H:%M')
train["dow"]=train["time"].dt.dayofweek
train["is_weekend"]=train["dow"].apply( lambda x : True if x  in [5,6] else False )
train["year"]=train["time"].dt.year
train["hour"]=train["time"].dt.hour
train["minute"]=train["time"].dt.minute
test["time"]=pd.to_datetime(test["time"],format='%Y-%m-%d %H:%M')
test["dow"]=test["time"].dt.dayofweek
test["weekend"]=test["dow"].apply( lambda x : True if x  in [5,6] else False )
test["year"]=test["time"].dt.year
test["hour"]=test["time"].dt.hour
test["minute"]=test["time"].dt.minute
mean_hour=train.groupby("hour").number_of_ticket.mean().rename("mean_hour").reset_index()
train=train.merge(mean_hour,how="left",on="hour")
test=test.merge(mean_hour,how="left",on="hour")
test["mean_hour"].fillna(mean_hour.mean_hour.mean(),inplace=True)
mean_travel_from=train.groupby("travel_from").number_of_ticket.mean().rename("mean_travel_from").reset_index()
train=train.merge(mean_travel_from,how="left",on="travel_from")
test=test.merge(mean_travel_from,how="left",on="travel_from")
test["mean_travel_from"].fillna(mean_travel_from.mean_travel_from.mean(),inplace=True)
mean_minute=train.groupby("minute").number_of_ticket.mean().rename("mean_minute").reset_index()
train=train.merge(mean_minute,how="left",on="minute")
test=test.merge(mean_minute,how="left",on="minute")
test["mean_minute"].fillna(mean_minute.mean_minute.mean(),inplace=True)
mean_hour_dow=train.groupby(["hour","dow"]).number_of_ticket.mean().rename("mean_hour_dow").reset_index()
train=train.merge(mean_hour_dow,how="left",on=["hour","dow"])
test=test.merge(mean_hour_dow,how="left",on=["hour","dow"])
test["mean_hour_dow"].fillna(mean_hour_dow.mean_hour_dow.mean(),inplace=True)
mean_hour_travel_from=train.groupby(["hour","travel_from"]).number_of_ticket.mean().rename("mean_hour_travel_from").reset_index()
train=train.merge(mean_hour_travel_from,how="left",on=["hour","travel_from"])
test=test.merge(mean_hour_travel_from,how="left",on=["hour","travel_from"])
test["mean_hour_travel_from"].fillna(mean_hour_travel_from.mean_hour_travel_from.mean(),inplace=True)
mean_dow_travel_from=train.groupby(["dow","travel_from"]).number_of_ticket.mean().rename("mean_dow_travel_from").reset_index()
train=train.merge(mean_dow_travel_from,how="left",on=["dow","travel_from"])
test=test.merge(mean_dow_travel_from,how="left",on=["dow","travel_from"])
test["mean_dow_travel_from"].fillna(mean_dow_travel_from.mean_dow_travel_from.mean(),inplace=True)
mean_minute_travel_from=train.groupby(["minute","travel_from"]).number_of_ticket.mean().rename("mean_minute_travel_from").reset_index()
train=train.merge(mean_minute_travel_from,how="left",on=["minute","travel_from"])
test=test.merge(mean_minute_travel_from,how="left",on=["minute","travel_from"])
test["mean_minute_travel_from"].fillna(mean_minute_travel_from.mean_minute_travel_from.mean(),inplace=True)
train_id=train.ride_id.unique().tolist()
test_id=test.ride_id.unique().tolist()
full_data=pd.concat([test,train],sort=True)
ride_distance={'Awendo': 351, 'Homa Bay': 364, 'Kehancha': 387, 'Kendu Bay': 347, 'Keroka': 281, 'Keumbu': 295, 'Kijauri': 271, 'Kisii': 305.1, 'Mbita': 401,
 'Migori': 372, 'Ndhiwa': 371, 'Nyachenge': 326, 'Oyugis': 330, 'Rodi': 348, 'Rongo': 332, 'Sirare': 392, 'Sori': 399}
full_data["ride_distance"]=full_data.travel_from.map(ride_distance)
ride_duration={'Awendo': 398,
 'Homa Bay': 420, 'Kehancha': 430, 'Kendu Bay': 370, 'Keroka': 300, 'Keumbu': 320, 'Kijauri': 290,
 'Kisii': 334, 'Mbita': 443, 'Migori': 428, 'Ndhiwa': 420, 'Nyachenge': 370, 'Oyugis': 350, 'Rodi': 400, 'Rongo': 381, 
'Sirare': 450, 'Sori': 450}
full_data["ride_duration"]=full_data.travel_from.map(ride_duration)
map_dict={30:[473, 485, 2363, 3614, 4981],31:[14648, 14893, 15110],19:[13843, 13893, 13985, 14832, 14980],38:[13842, 13960, 14584, 15242],2:[14051, 14457, 15250],
        26:[13848, 13892, 13968, 14653, 14734],27:[14061, 14147, 14311, 14437, 14594, 14660, 15302],42:[14146, 15004, 15231],10:[14115, 14190, 14295, 14530, 14969, 15201, 15267],
        45:[13887, 13956, 14041, 14423, 14651, 14720, 14884],42:[14146, 15004, 15231]}
ride_duration_time={ key: timedelta( minutes=value) for key ,value in zip(ride_duration.keys(),ride_duration.values())}
full_data["ride_duration_time"]=full_data.travel_from.map(ride_duration_time)
full_data["time_to_Nairoubi"]=full_data.time+ full_data.ride_duration_time
full_data["hour_to_Nairoubi"]=full_data["time_to_Nairoubi"].dt.hour
full_data["rush_hour_in_Nairoubi"]=0
full_data.loc[full_data.hour_to_Nairoubi.between(8,18),"rush_hour_in_Nairoubi"]=1
del full_data["ride_duration_time"],full_data["time_to_Nairoubi"],full_data["hour_to_Nairoubi"]

full_data["count_trip_travel_from_1"]=full_data.groupby([pd.Grouper(key="time",freq='1min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_1"]=full_data.groupby([pd.Grouper(key="time",freq='1min')]).ride_id.transform("count")

full_data["count_trip_travel_from_2"]=full_data.groupby([pd.Grouper(key="time",freq='2min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_per_2"]=full_data.groupby([pd.Grouper(key="time",freq='2min')]).ride_id.transform("count")

full_data["Date"]=full_data.time.dt.date
full_data["count_trip_per_day"]=full_data.groupby("Date").ride_id.transform("count")
full_data["count_trip_per_day_travel_from"]=full_data.groupby(["Date","travel_from"]).ride_id.transform("count")

full_data.sort_values("time",inplace=True)
full_data["count_trip_per_day_yesterday"]=full_data.groupby("Date").count_trip_per_day.shift(1)
full_data["count_trip_per_day_tommorw"]=full_data.groupby("Date").count_trip_per_day.shift(-1)
fea=["count_trip_per_day_yesterday","count_trip_per_day_tommorw"]
full_data[fea]=full_data[fea].fillna(method="ffill")
full_data[fea]=full_data[fea].fillna(method="backfill")
del full_data["Date"]
full_data["count_trip_per_month"]=full_data.groupby([pd.Grouper(key="time",freq='M')]).ride_id.transform("count")
full_data["count_trip_travel_from_5"]=full_data.groupby([pd.Grouper(key="time",freq='5min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_5"]=full_data.groupby([pd.Grouper(key="time",freq='5min')]).ride_id.transform("count")

full_data["count_trip_travel_from_8"]=full_data.groupby([pd.Grouper(key="time",freq='8min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_8"]=full_data.groupby([pd.Grouper(key="time",freq='8min')]).ride_id.transform("count")
full_data=full_data.sort_values(["travel_from","time"])
full_data["time_next_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(-1)).dt.total_seconds()/360
full_data["time_last_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(1)).dt.total_seconds()/360
full_data["time_next_next_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(-2)).dt.total_seconds()/360
full_data["time_last_last_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(2)).dt.total_seconds()/360
full_data["time_next_next_next_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(-3)).dt.total_seconds()/360
full_data["time_last_last_last_ride"]=(full_data["time"]-full_data.groupby(["travel_from"]).time.shift(3)).dt.total_seconds()/360
colums=['time_next_ride', 'time_last_ride', 'time_next_next_ride', 'time_last_last_ride',
     'time_next_next_next_ride', 'time_last_last_last_ride']
full_data[colums]=full_data.groupby(["travel_from"])[colums].fillna(method="ffill")
full_data[colums]=full_data.groupby(["travel_from"])[colums].fillna(method="backfill")
full_data["count_travel_from"]=full_data.travel_from.map(full_data.travel_from.value_counts())
full_data["count_travel_from"]=pd.cut(full_data["count_travel_from"],35,labels=False)
full_data["count_trip_travel_from_10"]=full_data.groupby([pd.Grouper(key="time",freq='10min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_10"]=full_data.groupby([pd.Grouper(key="time",freq='10min')]).ride_id.transform("count")
full_data["count_trip_travel_from_15"]=full_data.groupby([pd.Grouper(key="time",freq='15min'),"travel_from"]).ride_id.transform("count")
full_data["count_trip_15"]=full_data.groupby([pd.Grouper(key="time",freq='15min')]).ride_id.transform("count")
from sklearn import preprocessing
full_data["car_type"]=preprocessing.LabelEncoder().fit_transform(full_data["car_type"])
full_data["year"]=preprocessing.LabelEncoder().fit_transform(full_data["year"])
full_data["travel_from"]=preprocessing.LabelEncoder().fit_transform(full_data["travel_from"])
full_data["weekend"]=preprocessing.LabelEncoder().fit_transform(full_data["weekend"])
full_data["month"]=full_data.time.dt.month
uber_data=pd.read_csv("./Data/uber_data_monthly.csv")
full_data=full_data.merge(uber_data,how="left",on="month")
train=full_data[full_data.ride_id.isin(train_id)]
test=full_data[full_data.ride_id.isin(test_id)]
not_in_train=["count_trip_per_day_yesterday","count_trip_per_day_tommorw","count_trip_per_month","count_travel_from","count_trip_per_day",
#               "max_capacity","hour","dow","car_type","travel_from",
              "time","number_of_ticket","ride_id","travel_date","travel_time","count_trip_per_day_travel_from","count_trip_per_day_travel_from"]
features_name=train.drop(not_in_train,axis=1).columns
X_train=train[features_name].values
X_test=test[features_name].values
X_target=train.number_of_ticket.values
id_test=test.ride_id

from sklearn.model_selection import KFold
import xgboost as xgb
K = 10
kf = KFold(n_splits = K, random_state = 2019, shuffle = True)
d_test = xgb.DMatrix(X_test,feature_names=features_name)
valid=np.zeros_like(X_target)
xgb_pred=np.zeros(len(test))
for train_index, test_index in kf.split(train):
   
    train_X, valid_X = X_train[train_index,:], X_train[test_index,:]
    train_y, valid_y = X_target[train_index], X_target[test_index]

    xgb_params = {'min_child_weight': 10, 'eta': 0.004, 'colsample_bytree': 0.7, 'max_depth': 9,
            'subsample': 0.9, 'lambda': 5, 'nthread': 8, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear',"gamma":0.4 ,"alpha":0.02}
    d_train = xgb.DMatrix(train_X, train_y,feature_names=features_name)
    d_valid = xgb.DMatrix(valid_X, valid_y,feature_names=features_name)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000,  watchlist, maximize=False, verbose_eval=50, early_stopping_rounds=50)
    valid[test_index] =model.predict(d_valid)                
    xgb_pred = xgb_pred+model.predict(d_test)/(K)

print(mean_absolute_error(X_target,valid))

xgb_pred=np.round(xgb_pred)
model_final=pd.DataFrame({"ride_id":test.ride_id.tolist(),"number_of_ticket":xgb_pred})
for K,v in zip(map_dict.keys(),map_dict.values()):
    model_final.loc[model_final.ride_id.isin(v),"number_of_ticket"]=K
model_final=model_final[["ride_id","number_of_ticket"]]
model_final.to_csv("submission.csv",index=False)

