# %% [markdown] {"id":"bdfbc4dd"}
# #### Import Libraries

# %% [code] {"id":"210ce524","execution":{"iopub.status.busy":"2022-09-07T12:53:04.593459Z","iopub.execute_input":"2022-09-07T12:53:04.593983Z","iopub.status.idle":"2022-09-07T12:53:07.253467Z","shell.execute_reply.started":"2022-09-07T12:53:04.593906Z","shell.execute_reply":"2022-09-07T12:53:07.251553Z"}}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import itertools
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler,MinMaxScaler
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
# import lightgbm as lgb
import gc

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import metrics
import xgboost as xgb


# %% [markdown] {"id":"0cjdIcdQrFAN"}
# #Load Datasets:

# %% [code] {"execution":{"iopub.status.busy":"2022-09-07T12:53:07.255716Z","iopub.execute_input":"2022-09-07T12:53:07.256111Z","iopub.status.idle":"2022-09-07T12:53:07.261490Z","shell.execute_reply.started":"2022-09-07T12:53:07.256078Z","shell.execute_reply":"2022-09-07T12:53:07.260512Z"}}
path = ''

# %% [code] {"id":"6247095e","execution":{"iopub.status.busy":"2022-09-07T12:53:07.262983Z","iopub.execute_input":"2022-09-07T12:53:07.263632Z","iopub.status.idle":"2022-09-07T12:53:22.649604Z","shell.execute_reply.started":"2022-09-07T12:53:07.263592Z","shell.execute_reply":"2022-09-07T12:53:22.648307Z"}}
train = pd.read_csv(path+"Train.csv",parse_dates=['Date'])
test = pd.read_csv(path+"Test.csv",parse_dates=['Date'])
train_stats = pd.read_csv(path+"train_game_statistics.csv")
test_stats = pd.read_csv(path+"test_game_statistics.csv")

# %% [code] {"id":"S-rERS-VKgWg","execution":{"iopub.status.busy":"2022-09-07T12:53:22.653129Z","iopub.execute_input":"2022-09-07T12:53:22.653755Z","iopub.status.idle":"2022-09-07T12:53:23.019006Z","shell.execute_reply.started":"2022-09-07T12:53:22.653699Z","shell.execute_reply":"2022-09-07T12:53:23.017951Z"}}
def mean_X_change(stats):
    # Function to get the mean change of X axis by Team , to determine who is attacking and who is defending 
    ch = stats.groupby(['Game_ID','Team']).apply(lambda x: x.Player_ID.unique().shape[0]%11  )
    first_appear = stats.groupby(['Game_ID','Team','Player_ID']).apply(lambda x: x['Start_minutes'].min())
    mean_X = stats.groupby(['Game_ID','Team','Player_ID']).apply(lambda x:  x['X'].mean())
    new_df = first_appear.reset_index().merge(mean_X.reset_index(),on = ['Game_ID','Team','Player_ID'],how = 'left')
    new_df = new_df.groupby(['Game_ID','Team']).apply(lambda x : x.sort_values(by = '0_x',ascending = False)).reset_index(drop = True)
    df = ch.reset_index()[['Game_ID','Team']].copy()
    df[['mean_change_X' ]] =np.nan
    ch = ch.reset_index()
    for g in tqdm(ch.Game_ID.unique()):
        for t in ch.loc[ch.Game_ID == g,'Team']:
            i = ch.loc[(ch.Game_ID == g)&(ch.Team == t),0].values[0]
            df.loc[(df.Game_ID == g)&(df.Team == t),'mean_change_X'] = np.mean(new_df.loc[(new_df.Game_ID == g)&(new_df.Team == t),'0_y'].values[:i])
    
    return df
def mean_min_change(stats):
    # Function to get mean of minues where changes to the line-up occured
    ch = stats.groupby(['Game_ID','Team']).apply(lambda x: x.Player_ID.unique().shape[0]%11  )
    first_appear = stats.groupby(['Game_ID','Team','Player_ID']).apply(lambda x: x['Start_minutes'].min())
    mean_X = stats.groupby(['Game_ID','Team','Player_ID']).apply(lambda x:  x['X'].mean())
    new_df = first_appear.reset_index().merge(mean_X.reset_index(),on = ['Game_ID','Team','Player_ID'],how = 'left')
    new_df = new_df.groupby(['Game_ID','Team']).apply(lambda x : x.sort_values(by = '0_x',ascending = False)).reset_index(drop = True)
    df = ch.reset_index()[['Game_ID','Team']].copy()
    df[['mean_change_min' ]] =np.nan
    ch = ch.reset_index()
    for g in tqdm(ch.Game_ID.unique()):
        for t in ch.loc[ch.Game_ID == g,'Team']:
            i = ch.loc[(ch.Game_ID == g)&(ch.Team == t),0].values[0]
            df.loc[(df.Game_ID == g)&(df.Team == t),'mean_change_min'] = np.mean(new_df.loc[(new_df.Game_ID == g)&(new_df.Team == t),'0_x'].values[:i])
    
    return df
def get_stats(train,train_stats):
    # Function to get stats by team ( goals (scored and conceded) per game hoe and away) ## result in overfitting maybe should use kfold before encoding
    teams = train_stats.loc[train_stats['Goals_scored']==1]
    team_goals = teams.groupby(['Game_ID','Team'])['Goals_scored'].sum()
    team_goals =team_goals.reset_index()
    for game in train.Game_ID:
        s1 = train.loc[train.Game_ID == game,['Home Team','Away Team']].values
    #     team_goals.loc[team_goals.Game_ID == 'ID_00EFNL7L']
        s2 = team_goals.loc[team_goals.Game_ID == game,'Team'].values
        diff = np.setdiff1d(s1,s2)
        if diff.shape[0] == 1: 
            team_goals.loc[-1] = [game, diff[0], 0]  # adding a row
            team_goals.index = team_goals.index + 1  # shifting index
            team_goals = team_goals.sort_index()
    teams2 = train_stats.loc[train_stats['Goals_conceded']==1]
    team_goals2 = teams2.groupby(['Game_ID','Team'])['Goals_conceded'].sum()
    team_goals2 =team_goals2.reset_index()
    for game in train.Game_ID:
        s1 = train.loc[train.Game_ID == game,['Home Team','Away Team']].values
    #     team_goals.loc[team_goals.Game_ID == 'ID_00EFNL7L']
        s2 = team_goals2.loc[team_goals2.Game_ID == game,'Team'].values
        diff = np.setdiff1d(s1,s2)
        if diff.shape[0] == 1: 
            team_goals2.loc[-1] = [game, diff[0], 0]  # adding a row
            team_goals2.index = team_goals2.index + 1  # shifting index
            team_goals2 = team_goals2.sort_index()  
    diff_score = team_goals2.groupby('Team')['Goals_conceded'].mean().reset_index().merge(team_goals.groupby('Team')['Goals_scored'].mean().reset_index(),on = 'Team' )
    ast =team_goals2.merge(team_goals,on = ['Game_ID','Team']).merge(train[['Game_ID','Home Team','Away Team']],on = 'Game_ID')
    as_home = ast.loc[ast['Team'] == ast['Home Team']].groupby('Team')[['Goals_scored','Goals_conceded']].mean().reset_index()
    as_away = ast.loc[ast['Team'] == ast['Away Team']].groupby('Team')[['Goals_scored','Goals_conceded']].mean().reset_index()
    as_home['diff_score_h'] = as_home['Goals_scored'] - as_home['Goals_conceded'] 
    as_away['diff_score_a'] = as_away['Goals_scored'] - as_away['Goals_conceded'] 
    as_home = as_home[['Team','diff_score_h']]
    as_away = as_away[['Team','diff_score_a']]
    diff_score = team_goals2.groupby('Team')['Goals_conceded'].mean().reset_index().merge(team_goals.groupby('Team')['Goals_scored'].mean().reset_index(),on = 'Team' )
    diff_score['diff_score'] = diff_score['Goals_scored'] - diff_score['Goals_conceded']
    diff_score = diff_score[['Team','diff_score']]
    diff_score = diff_score.merge(as_home,on = 'Team').merge(as_away,on = 'Team')
#     st = get_stats(train,train_stats)
    diff_score.index = diff_score.Team
    diff_score.drop('Team',axis = 1,inplace = True)
    return diff_score
def preprocess(train,test,train_stats,test_stats,scale = False,th = 2,encode = False):
    # modefied function from mkm notebook
    train['Train']=1
    test['Train']=0
    home_teams=train['Home Team'].unique()
    all_data=pd.concat([train,test])
    if encode:
        lb = LabelEncoder()
        all_data['Away_ID'] = lb.fit_transform(all_data['Away Team'].values.reshape(-1,1))
        all_data['Home_ID'] = lb.transform(all_data['Home Team'].values.reshape(-1,1))

    else:
        for team in home_teams:
            all_data['home_'+team]=0
            all_data.loc[all_data['Home Team']==team,'home_'+team]=1
        away_teams=train['Away Team'].unique()
        for team in away_teams:
            all_data['away_'+team]=0
            all_data.loc[all_data['Away Team']==team,'away_'+team]=1
#     print(all_data.isna().sum())
    all_data.fillna(all_data.mean(),inplace=True)
    # include month//day... maybe that can affect the presence of supporters which affect team performance
    all_data['month'] = all_data.Date.dt.month
    all_data['year'] = all_data.Date.dt.year
    all_data['day'] = all_data.Date.dt.day
    all_data['week'] = all_data.Date.dt.week
    all_data['dow'] = all_data.Date.dt.dayofweek
    all_data['woy'] = all_data.Date.dt.weekofyear
    train = all_data[all_data["Train"] == 1]
    test = all_data[all_data["Train"] == 0]
    train_inv= train.copy()
    train_inv['Home Team'] = train['Away Team']
    train_inv['Away Team'] = train['Home Team']
    # Last score is the score of the last game between the teams
    train['last_Score'] = 0
    test['last_Score'] = 0
    for game in tqdm(train.Game_ID):
        teamh = train.loc[train.Game_ID == game,'Home Team'].values[0]
        teama = train.loc[train.Game_ID == game,'Away Team'].values[0]
        date = train.loc[train.Game_ID == game,'Date'].values[0]
        seas = train.loc[train.Game_ID == game,'Season'].values[0]
        temp = train.loc[(train['Home Team'] == teamh) & (train['Away Team'] == teama),
                         ['Game_ID','Date','Season','Score'] ]
        temp_inv = train_inv.loc[(train_inv['Home Team'] == teamh) & (train_inv['Away Team'] == teama),
                         ['Game_ID','Date','Season','Score']]
        temp_inv['Score'] = temp_inv['Score'].map({0:2,2:0})
        if (temp.shape[0] ==0) & (temp_inv.shape[0] ==0):
            train.loc[train.Game_ID == game,'last_Score'] = -1 # No data
        elif (temp.shape[0] ==0):
            game_id = -1
            for g in temp_inv.Game_ID:
                if temp_inv.loc[temp_inv.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if game_id == -1:
                train.loc[train.Game_ID == game,'last_Score'] = -1 # No recent data
            else:
                train.loc[train.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id,'Score'].values[0]
        elif (temp_inv.shape[0] ==0):
            game_id = -1
            for g in temp.Game_ID:
                if temp.loc[temp.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if game_id == -1:
                train.loc[train.Game_ID == game,'last_Score'] = -1 # No recent data
            else:
                train.loc[train.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
        else:
            game_id_inv = -1
            for g in temp_inv.Game_ID:
                if temp_inv.loc[temp_inv.Game_ID == g,'Date'].values[0]<date:
                    game_id_inv = g
            game_id = -1
            for g in temp.Game_ID:
                if temp.loc[temp.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if (game_id == -1) & (game_id_inv == -1):
                train.loc[train.Game_ID == game,'last_Score'] = -1 # No data
            elif game_id ==-1:
                train.loc[train.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Score'].values[0]
            elif game_id_inv == -1:
                train.loc[train.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
            else:
                if temp.loc[temp.Game_ID == game_id,'Date'].values[0]> temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Date'].values[0]:
                    train.loc[train.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Score'].values[0]
                else:
                    train.loc[train.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
    for game in tqdm(test.Game_ID):
        teamh = test.loc[test.Game_ID == game,'Home Team'].values[0]
        teama = test.loc[test.Game_ID == game,'Away Team'].values[0]
        date = test.loc[test.Game_ID == game,'Date'].values[0]
        seas = test.loc[test.Game_ID == game,'Season'].values[0]
        temp = train.loc[(train['Home Team'] == teamh) & (train['Away Team'] == teama),
                         ['Game_ID','Date','Season','Score'] ]
        temp_inv = train_inv.loc[(train_inv['Home Team'] == teamh) & (train_inv['Away Team'] == teama),
                         ['Game_ID','Date','Season','Score']]
        temp_inv['Score'] = temp_inv['Score'].map({0:2,2:0})
        if (temp.shape[0] ==0) & (temp_inv.shape[0] ==0):
            test.loc[test.Game_ID == game,'last_Score'] = np.nan # No data
        elif (temp.shape[0] ==0):
            game_id = None
            for g in temp_inv.Game_ID:
                if temp_inv.loc[temp_inv.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if game_id == None:
                test.loc[test.Game_ID == game,'last_Score'] = np.nan # No recent data
            else:
                test.loc[test.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id,'Score'].values[0]
        elif (temp_inv.shape[0] ==0):
            game_id = None
            for g in temp.Game_ID:
                if temp.loc[temp.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if game_id == None:
                test.loc[test.Game_ID == game,'last_Score'] = np.nan # No recent data
            else:
                test.loc[test.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
        else:
            game_id_inv = None
            for g in temp_inv.Game_ID:
                if temp_inv.loc[temp_inv.Game_ID == g,'Date'].values[0]<date:
                    game_id_inv = g
            game_id = None
            for g in temp.Game_ID:
                if temp.loc[temp.Game_ID == g,'Date'].values[0]<date:
                    game_id = g
            if (game_id == None) & (game_id_inv == None):
                test.loc[test.Game_ID == game,'last_Score'] = np.nan # No data
            elif game_id ==None:
                test.loc[test.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Score'].values[0]
            elif game_id_inv == None:
                test.loc[test.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
            else:
                if temp.loc[temp.Game_ID == game_id,'Date'].values[0]> temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Date'].values[0]:
                    test.loc[test.Game_ID == game,'last_Score'] = temp_inv.loc[temp_inv.Game_ID == game_id_inv,'Score'].values[0]
                else:
                    test.loc[test.Game_ID == game,'last_Score'] = temp.loc[temp.Game_ID == game_id,'Score'].values[0]
    train['last_Score'] = train['last_Score'].fillna(0).astype('int')
    test['last_Score'] =  test['last_Score'].fillna(0).astype('int')
    all_stats = pd.concat([train_stats,test_stats])
    all_stats.drop(['Action', 'Goals_scored', 'Goals_conceded', 'next_action'],axis=1,inplace = True)
    all_stats.drop(['next_player','next_x','next_y','event_id','next_team','next_event_id','xt_value'],axis =1,inplace = True)
    # we should delete some rows from stats with X == 300 , Y ==300 , i think those are nan values 
    # then we calculate some game statistics like number of shots ; number of passes, accuracy (shots and passes) , 
    # For each half
    all_stats = all_stats[all_stats.X !=300]
    grouped = all_stats.groupby(['Game_ID','Team','Player_ID','Half'])[['Shots','SoT','Accurate passes','Inaccurate passes','Passes']].sum()
    grouped =grouped.reset_index()
    accu_shots = grouped.groupby(['Game_ID','Team','Half']).apply(lambda x : x['SoT'].sum()/x['Shots'].sum()).reset_index()
    accu_pass = grouped.groupby(['Game_ID','Team','Half']).apply(lambda x : x['Accurate passes'].sum()/x['Passes'].sum()).reset_index()
    n_shots = grouped.groupby(['Game_ID','Team','Half']).apply(lambda x : x['SoT'].sum()).reset_index()
    n_passes = grouped.groupby(['Game_ID','Team','Half']).apply(lambda x : x['Passes'].sum()).reset_index()
    train[['n_passes_home1','n_shots_home1','accu_pass_home1','accu_shots_home1',
          'n_passes_away1','n_shots_away1','accu_pass_away1','accu_shots_away1',
          'n_passes_home2','n_shots_home2','accu_pass_home2','accu_shots_home2',
          'n_passes_away2','n_shots_away2','accu_pass_away2','accu_shots_away2']]=0
    for game in tqdm(train.Game_ID):
        train.loc[train.Game_ID == game,'n_passes_home1'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (n_passes.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_passes_home2'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (n_passes.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_passes_away1'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (n_passes.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_passes_away2'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (n_passes.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_shots_home1'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (n_shots.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_shots_home2'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (n_shots.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_shots_away1'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (n_shots.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'n_shots_away2'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (n_shots.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_pass_home1'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (accu_pass.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_pass_home2'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (accu_pass.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_pass_away1'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (accu_pass.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_pass_away2'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (accu_pass.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_shots_home1'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (accu_shots.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_shots_home2'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == train.loc[train.Game_ID == game,'Home Team'].values[0]) & (accu_shots.Half == '2nd half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_shots_away1'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (accu_shots.Half == '1st half' ),0].values[0]
        train.loc[train.Game_ID == game,'accu_shots_away2'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == train.loc[train.Game_ID == game,'Away Team'].values[0]) & (accu_shots.Half == '2nd half' ),0].values[0]
    test[['n_passes_home1','n_shots_home1','accu_pass_home1','accu_shots_home1',
          'n_passes_away1','n_shots_away1','accu_pass_away1','accu_shots_away1',
          'n_passes_home2','n_shots_home2','accu_pass_home2','accu_shots_home2',
          'n_passes_away2','n_shots_away2','accu_pass_away2','accu_shots_away2']]=0
    exep = []
    for game in tqdm(test.Game_ID):
        try:
            test.loc[test.Game_ID == game,'n_passes_home1'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (n_passes.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_passes_home2'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (n_passes.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue

        try:
            test.loc[test.Game_ID == game,'n_passes_away1'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (n_passes.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_passes_away2'] = n_passes.loc[(n_passes.Game_ID == game)&(n_passes.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (n_passes.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_shots_home1'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (n_shots.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_shots_home2'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (n_shots.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_shots_away1'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (n_shots.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'n_shots_away2'] = n_shots.loc[(n_shots.Game_ID == game)&(n_shots.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (n_shots.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_pass_home1'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (accu_pass.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_pass_home2'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (accu_pass.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_pass_away1'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (accu_pass.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_pass_away2'] = accu_pass.loc[(accu_pass.Game_ID == game)&(accu_pass.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (accu_pass.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_shots_home1'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (accu_shots.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_shots_home2'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == test.loc[test.Game_ID == game,'Home Team'].values[0]) & (accu_shots.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_shots_away1'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (accu_shots.Half == '1st half' ),0].values[0]
        except:
            exep.append(game)
            continue
        try:
            test.loc[test.Game_ID == game,'accu_shots_away2'] = accu_shots.loc[(accu_shots.Game_ID == game)&(accu_shots.Team == test.loc[test.Game_ID == game,'Away Team'].values[0]) & (accu_shots.Half == '2nd half' ),0].values[0]
        except:
            exep.append(game)
            continue
    # train
# train[train.Game_ID == 'ID_00EFNL7L']
# train
#     th= 2
    test_stats = test_stats[test_stats.X<300]
    train_stats['dist_cent'] = np.sqrt((train_stats.X - 52.5)**2 + (train_stats.Y - 34)**2) 
    a = train_stats[train_stats['Accurate passes']==1].groupby(['Game_ID','Team']).apply(lambda x : (x['dist_cent']<th).sum())
    b = a.reset_index().copy()
    ch = train_stats.groupby(['Game_ID','Team']).apply(lambda x: x.Player_ID.unique().shape[0]%11)
    ch = ch.reset_index()
    change_mean_X = mean_X_change(train_stats)
    change_mean_min = mean_min_change(train_stats)
    c =train_stats.groupby(['Game_ID','Team','End_minutes']).apply(lambda x: x['Start_minutes'].min()).reset_index()
    c =c.groupby(['Game_ID','Team']).apply(lambda x: (x['End_minutes']-x[0]).mean()).reset_index()
    # Number of chnges per team , mean of X per team, sum of minutes per Team (proxy to possession maybe)
    
    # Number of centers played by a team since the team who plays most centers loses ; but i guess the actual 
    # center after the goal was removed so i tried to locate the next pass , with few trials i decided to take a 
    # threshhold of 2 .
    train['n_changes_home'] = 0
    train['n_changes_away'] = 0
    train['mean_X_home'] = 0
    train['mean_X_away'] = 0
    train['change_mean_X_home'] = 0
    train['change_mean_X_away'] = 0
    train['change_mean_min_home'] = 0
    train['change_mean_min_away'] = 0
    train['sum_min_home'] = 0
    train['sum_min_away'] = 0
    mean_X = train_stats.groupby(['Game_ID','Team']).apply(lambda x : x['X'].mean()).reset_index()
    # 
    train['h_cent'] = 0
    train['a_cent'] = 0
    for g in tqdm(train.Game_ID):
        home = train.loc[train.Game_ID == g ,'Home Team'].values[0]
        away = train.loc[train.Game_ID == g ,'Away Team'].values[0]
        train.loc[train.Game_ID == g,'h_cent'] = b.loc[(b.Game_ID == g)&(b.Team == home),0].values[0]
        train.loc[train.Game_ID == g,'a_cent'] = b.loc[(b.Game_ID == g)&(b.Team == away),0].values[0]
        train.loc[train.Game_ID == g,'mean_X_home'] = mean_X.loc[(mean_X.Game_ID == g)&(mean_X.Team == home),0].values[0]
        train.loc[train.Game_ID == g,'mean_X_away'] = mean_X.loc[(mean_X.Game_ID == g)&(mean_X.Team == away),0].values[0]
        train.loc[train.Game_ID == g,'n_changes_home'] = ch.loc[(ch.Game_ID == g)&(ch.Team == home),0].values[0]
        train.loc[train.Game_ID == g,'n_changes_away'] = ch.loc[(ch.Game_ID == g)&(ch.Team == away),0].values[0]
        train.loc[train.Game_ID == g,'change_mean_X_home'] = change_mean_X.loc[(change_mean_X.Game_ID == g)&(change_mean_X.Team == home),'mean_change_X'].values[0]
        train.loc[train.Game_ID == g,'change_mean_X_away'] = change_mean_X.loc[(change_mean_X.Game_ID == g)&(change_mean_X.Team == away),'mean_change_X'].values[0]
        train.loc[train.Game_ID == g,'change_mean_min_home'] = change_mean_min.loc[(change_mean_min.Game_ID == g)&(change_mean_min.Team == home),'mean_change_min'].values[0]
        train.loc[train.Game_ID == g,'change_mean_min_away'] = change_mean_min.loc[(change_mean_min.Game_ID == g)&(change_mean_min.Team == away),'mean_change_min'].values[0]
        train.loc[train.Game_ID == g,'sum_min_home'] = c.loc[(c.Game_ID == g)&(c.Team == home),0].values[0]
        train.loc[train.Game_ID == g,'sum_min_away'] = c.loc[(c.Game_ID == g)&(c.Team == away),0].values[0]
    test_stats['dist_cent'] = np.sqrt((test_stats.X - 52.5)**2 + (test_stats.Y - 34)**2) 
    a = test_stats[test_stats['Accurate passes']==1].groupby(['Game_ID','Team']).apply(lambda x : (x['dist_cent']<th).sum())
    b = a.reset_index().copy()
    mean_X = test_stats.groupby(['Game_ID','Team']).apply(lambda x : x['X'].mean()).reset_index()
    ch = test_stats.groupby(['Game_ID','Team']).apply(lambda x: x.Player_ID.unique().shape[0]%11)
    ch = ch.reset_index()
    change_mean_X = mean_X_change(test_stats)
    c =test_stats.groupby(['Game_ID','Team','End_minutes']).apply(lambda x: x['Start_minutes'].min()).reset_index()
    c =c.groupby(['Game_ID','Team']).apply(lambda x: (x['End_minutes']-x[0]).mean()).reset_index()
    change_mean_min = mean_min_change(test_stats)

    test['change_mean_X_home'] = 0
    test['change_mean_X_away'] = 0
    test['h_cent'] = 0
    test['a_cent'] = 0
    test['n_changes_home'] = 0
    test['n_changes_away'] = 0
    test['mean_X_home'] = 0
    test['mean_X_away'] = 0
    test['change_mean_min_home'] = 0
    test['change_mean_min_away'] = 0
    test['sum_min_home'] = 0
    test['sum_min_away'] = 0
    for g in tqdm(test.Game_ID):
        home = test.loc[test.Game_ID == g ,'Home Team'].values[0]
        away = test.loc[test.Game_ID == g ,'Away Team'].values[0]
        try:
            test.loc[test.Game_ID == g,'mean_X_home'] = mean_X.loc[(mean_X.Game_ID == g)&(mean_X.Team == home),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'mean_X_away'] = mean_X.loc[(mean_X.Game_ID == g)&(mean_X.Team == away),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'n_changes_home'] = ch.loc[(ch.Game_ID == g)&(ch.Team == home),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'n_changes_away'] = ch.loc[(ch.Game_ID == g)&(ch.Team == away),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'change_mean_X_home'] = change_mean_X.loc[(change_mean_X.Game_ID == g)&(change_mean_X.Team == home),'mean_change_X'].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'change_mean_X_away'] = change_mean_X.loc[(change_mean_X.Game_ID == g)&(change_mean_X.Team == away),'mean_change_X'].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'sum_min_home'] = c.loc[(c.Game_ID == g)&(c.Team == home),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'sum_min_away'] = c.loc[(c.Game_ID == g)&(c.Team == away),0].values[0]
        except:
            continue

        try:
            test.loc[test.Game_ID == g,'h_cent'] = b.loc[(b.Game_ID == g)&(b.Team == home),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'a_cent'] = b.loc[(b.Game_ID == g)&(b.Team == away),0].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'change_mean_min_home'] = change_mean_min.loc[(change_mean_min.Game_ID == g)&(change_mean_min.Team == home),'mean_change_min'].values[0]
        except:
            continue
        try:
            test.loc[test.Game_ID == g,'change_mean_min_away'] = change_mean_min.loc[(change_mean_min.Game_ID == g)&(change_mean_min.Team == away),'mean_change_min'].values[0]
        except:
            continue
    st = get_stats(train,train_stats)
    train['diff_score'] = train['Home Team'].map(st['diff_score'])
    train['diff_score_h'] = train['Home Team'].map(st['diff_score_h'])
    train['diff_score_h'] = train['Home Team'].map(st['diff_score_h'])
    train['diff_score1'] = train['Away Team'].map(st['diff_score'])
    train['diff_score_h1'] = train['Away Team'].map(st['diff_score_h'])
    train['diff_score_h1'] = train['Away Team'].map(st['diff_score_h'])
    test['diff_score'] = test['Home Team'].map(st['diff_score'])
    test['diff_score_h'] = test['Home Team'].map(st['diff_score_h'])
    test['diff_score_h'] = test['Home Team'].map(st['diff_score_h'])
    test['diff_score1'] = test['Away Team'].map(st['diff_score'])
    test['diff_score_h1'] = test['Away Team'].map(st['diff_score_h'])
    test['diff_score_h1'] = test['Away Team'].map(st['diff_score_h'])
    
    train.fillna(0,inplace = True)# grouped.loc[(grouped.Game_ID == game)]
    test.fillna(0,inplace = True)
    train_cols = train.columns.difference(['Date', 'Season','Match_ID', 'Game_ID','Score','Train','Home Team','Away Team'])
    if scale:
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        scal = ['n_passes_away1',
        'n_passes_away2',       
        'n_passes_home1',        
        'n_passes_home2',        
        'n_shots_away1',
        'n_shots_away2',
        'n_shots_home1',
        'n_shots_home2']
        for s in scal:
            sc = MinMaxScaler()
            train[s] = sc.fit_transform(train[s].values.reshape(-1, 1))
            test[s] = sc.transform(test[s].values.reshape(-1, 1))
    if encode:
        return train[train_cols],test[train_cols],train['Score'],test['Score'],lb
    else:
        return train[train_cols],test[train_cols],train['Score'],test['Score']
#     return train , test
        # test.loc[test.Game_ID =='ID_00EFNL7L']
            # train.loc[train.Game_ID =='ID_00EFNL7L']

def _get_X_Y_DF_from_CV(train_X, train_Y, train_index, validation_index):
    # get X and Y given index
        X_train, X_validation = (
            train_X.iloc[train_index],
            train_X.iloc[validation_index],
        )
        y_train, y_validation = (
            train_Y.iloc[train_index],
            train_Y.iloc[validation_index],
        )
        return X_train, X_validation, y_train, y_validation
def scale_fea(X, test_data):
    # scale pretermined features
    scale = ['change_mean_X_home', 'change_mean_min_away', 'change_mean_min_home',
           'mean_X_away', 'mean_X_home',
             'n_passes_away1', 'n_passes_away2', 'n_passes_home1',
           'n_passes_home2', 'n_shots_away1', 'n_shots_away2',
           'n_shots_home1', 'n_shots_home2']
    # X[scale]
    sc = MinMaxScaler()
    X_scaled = X.copy()
    test_scaled = test_data.copy()
    X_scaled[scale] = sc.fit_transform(X[scale])
    test_scaled[scale] = sc.transform(test_data[scale])
    return X_scaled,test_scaled
def train_lgb(X,y,test_X,params):
    # function to train lightgbm given params using kfold
    features_importance= pd.DataFrame({'Feature':[], 'Importance':[]})
    models =[]
    train_X = X.copy()
    train_Y = y.copy()
#     test_X = test_data.copy()
    print(f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}")

    predictors = list(train_X.columns)
    # print(f"List of features to be used {list(predictors)}")

    # Selecting n_splits to be 3, since class 42 has 
    # just 3 instances
    kf = KFold(random_state=42,n_splits=K_FOLDS, shuffle=True)
    y_oof_lgb = np.zeros(shape=(len(train_X), NUM_CLASSES))
    y_predicted_lgb = np.zeros(shape=(len(test_X), NUM_CLASSES))
    cv_scores = []
    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X, y=train_Y):
        fold += 1
        print(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_DF_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=100,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            num_boost_round=N_ESTIMATORS,
            feature_name=predictors,
            categorical_feature="auto",
        )
        del lgb_train, lgb_eval, train_index, X_train, y_train
        gc.collect()

        y_oof_lgb[validation_index] = model.predict(
            X_validation, num_iteration=model.best_iteration
        )

        y_predicted_lgb += model.predict(
            test_X.values, num_iteration=model.best_iteration
        )
        fold_importance_df= pd.DataFrame({'Feature':[], 'Importance':[]})
        fold_importance_df['Feature']= predictors
        fold_importance_df['Importance']= model.feature_importance()
        fold_importance_df["fold"] = fold + 1
        features_importance = pd.concat([features_importance, fold_importance_df], axis=0)
        models.append(model)

        best_iteration = model.best_iteration
        print(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = metrics.log_loss(y_validation, y_oof_lgb[validation_index])
        cv_scores.append(cv_oof_score)
        print(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

    y_predicted_lgb /= n_folds
    oof_score = round(metrics.log_loss(train_Y, y_oof_lgb), 5)
    avg_cv_scores = round(sum(cv_scores) / len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)
    return y_predicted_lgb,models,y_oof_lgb,oof_score,features_importance
def train_catbo(train_X, train_Y,test_X,params):
# Function to train Catboost , using Kfold
    kf = KFold(random_state=42,n_splits=K_FOLDS, shuffle=True)
    y_oof = np.zeros(shape=(len(train_X), NUM_CLASSES))
    y_predicted = np.zeros(shape=(len(test_X), NUM_CLASSES))
    cv_scores = []
    models = []
    fold = 0
    n_folds = kf.get_n_splits()
    for train_index, validation_index in kf.split(X=train_X, y=train_Y):
        fold += 1
        print(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = _get_X_Y_DF_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        train_pool = Pool(data=X_train, label=y_train)
        eval_pool = Pool(data=X_validation, label=y_validation.values) 
        model = CatBoostClassifier(**params)
        model.fit(train_pool,plot=True,eval_set=eval_pool)
        del train_index, X_train, y_train
        gc.collect()
        models.append(model)
        y_oof[validation_index] = model.predict_proba(
            X_validation )

        y_predicted += model.predict_proba(
            test_X.values
        )

    #     best_iteration = model.best_iteration
    #     print(f"Best number of iterations for fold {fold} is: {best_iteration}")

        cv_oof_score = metrics.log_loss(y_validation, y_oof[validation_index])
        cv_scores.append(cv_oof_score)
        print(f"CV OOF Score for fold {fold} is {cv_oof_score}")

        del validation_index, X_validation, y_validation
        gc.collect()

    y_predicted /= n_folds
    oof_score = round(metrics.log_loss(train_Y, y_oof), 5)
    avg_cv_scores = round(sum(cv_scores) / len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)
    return y_predicted,models,y_oof,oof_score
def train_xgb(X,y,test_data,params,num_iter=1500,es = 100,ve = 100):
    # Function to train XGBoost given params using kfold
    features = X.columns
    X = X.values
    # y = train['target'].values
    y_oof = np.zeros(shape=(len(X), NUM_CLASSES))
    y_predicted = np.zeros(shape=(len(test_data), NUM_CLASSES))
    cv_scores = []
    models = []
    kf = KFold(random_state=SEED,n_splits=K_FOLDS, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(' xgb kfold: {}  of  {} : '.format(i+1, K_FOLDS ))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        d_train = xgb.DMatrix(X_train, y_train) 
        d_valid = xgb.DMatrix(X_valid, y_valid) 
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        xgb_model = xgb.train(params, d_train, num_iter, watchlist,
                              early_stopping_rounds=es, 
                            verbose_eval=ve)
        models.append(xgb_model)
        y_oof[test_index] = xgb_model.predict(xgb.DMatrix(X_valid), 
                            ntree_limit=xgb_model.best_ntree_limit)
        y_predicted += xgb_model.predict(xgb.DMatrix(test_data[features].values), 
                            ntree_limit=xgb_model.best_ntree_limit) 
        
        cv_oof_score = metrics.log_loss(y_valid, y_oof[test_index])
        cv_scores.append(cv_oof_score)
        print(f"CV OOF Score for fold {i+1} is {cv_oof_score}")

#         del validation_index, X_validation, y_validation
#         gc.collect()

    y_predicted /= K_FOLDS
    oof_score = round(metrics.log_loss(y, y_oof), 5)
    avg_cv_scores = round(sum(cv_scores) / len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)
    return y_predicted,models,y_oof,oof_score 

def train_keras(X,train,test_data):
    # train Keras model using Kfold
    train ['Original_score'] = le.inverse_transform(train.Score)
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X_scaled = scaler.fit_transform(X)
#     X_scaled = pd.DataFrame(X_scaled)/
#     test_scaled = scaler.transform(test_data)
#     test_scaled = pd.DataFrame(test_scaled)
    Y = pd.get_dummies(train['Original_score'])
    X_scaled = X.values
    test_scaled = test_data.copy()
    Y = Y.values
    inp = len(X.columns)
    y_oof = np.zeros(shape=(len(X), NUM_CLASSES))
    y_predicted = np.zeros(shape=(len(test_data), NUM_CLASSES))
    cv_scores = []
    models = []
    kf = KFold(random_state=SEED,n_splits=K_FOLDS, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X_scaled, Y)):
        print(' keras kfold: {}  of  {} : '.format(i+1, K_FOLDS ))
        X_train, X_valid = X_scaled[train_index], X_scaled[test_index]
        y_train, y_valid = Y[train_index], Y[test_index]
#         np.random.seed(SEED)
        my_model = baseline_model(inp)
        my_model.fit(X_train, y_train,
                     validation_data=(X_valid, y_valid),
                     epochs=1000,
                     callbacks=[EarlyStopping(patience=20)],
                     verbose=0)
        
        models.append(my_model)
        y_oof[test_index] = my_model.predict(X_valid)
        y_predicted += my_model.predict(test_scaled.values) 
        del my_model
        gc.collect()
        cv_oof_score = metrics.log_loss(y_valid, y_oof[test_index])
        cv_scores.append(cv_oof_score)
        print(f"CV OOF Score for fold {i+1} is {cv_oof_score}")

#         del validation_index, X_validation, y_validation
#         gc.collect()

    y_predicted /= K_FOLDS
    oof_score = round(metrics.log_loss(y, y_oof), 5)
    avg_cv_scores = round(sum(cv_scores) / len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)
    return y_predicted,models,y_oof,oof_score 


def preds_to_sub(test,le,y_predicted,save = False):
    # Turn predictrion to submission
    Test = test.copy()
    cols=le.inverse_transform([*range(3)])
    Test[cols]= y_predicted
    submit = Test[["Game_ID",'Away win', 'Draw', 'Home Win']]
    submit.drop_duplicates(subset = ["Game_ID"], inplace=True)
    submit = submit.reset_index(drop=True)
    if save:
        submit.to_csv("submission.csv", index=False)
    return submit   

# %% [code] {"execution":{"iopub.status.busy":"2022-09-07T12:53:23.020714Z","iopub.execute_input":"2022-09-07T12:53:23.021409Z","iopub.status.idle":"2022-09-07T12:53:35.584390Z","shell.execute_reply.started":"2022-09-07T12:53:23.021371Z","shell.execute_reply":"2022-09-07T12:53:35.583024Z"}}
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
# Keras imports
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping
import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 2021

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# tensorflow random seed 
import tensorflow as tf 
def seedTF(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)
    
# torch random seed
import torch
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)
seedEverything(42)
seed = 42

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)[:10].index
    best_features = feature_importance_df_[["Feature", "Importance"]].groupby("Feature").mean().sort_values(by="Importance", ascending=False)[:50]
    best_features.reset_index(inplace=True)
    print(best_features.dtypes)
    plt.figure(figsize=(8, 10))
    sns.barplot(x="Importance", y="Feature", data=best_features)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()


le=LabelEncoder()
train["Score"] = le.fit_transform(train["Score"])
score_mapping = dict(zip(le.classes_, range(len(le.classes_))))

train.sort_values(by=['Date'],inplace=True)

# prerpocess the data
X,test_data,y,_=preprocess(train,test,train_stats,test_stats,encode = False)


# since these columns tend to overfit
drop_fea = ['diff_score','diff_score_h','diff_score_h',
            'diff_score1','diff_score_h1','diff_score_h1']


SEED = 6
NUM_CLASSES = 3
EARLY_STOPPING_ROUNDS = 100
N_ESTIMATORS = 10000
K_FOLDS = 4
# Define Parameters for LGBM
lgb_params = {
    "objective": "multiclass",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_class": NUM_CLASSES,
    "num_leaves": 42,
#     "tree_learner": "serial",
    "n_jobs": 4,
    "seed": SEED,
    "max_depth": -1,
#     "max_bin": 255,
#     'reg_alpha': 0.1,  # L1 regularization term on weights
#     'reg_lambda': 1,
    "metric": "multi_logloss",
    "verbose": -1,
}
y_predicted_lgb,models_lgb,y_oof_lgb,oof_score_lgb,features_importance_lgb = train_lgb(X.drop(drop_fea,axis = 1),
                                                                                       y,test_data.drop(drop_fea,axis = 1),lgb_params)

from catboost import Pool, CatBoostClassifier
params_cat = {'iterations':1500,
        'learning_rate':0.01,
        'random_strength':0.1,
        'depth':8,
        'loss_function':'MultiClass',
        'eval_metric':'MultiClass',
        'verbose' : 100,
        'leaf_estimation_method':'Newton'}
y_predicted_cat,models_cat,y_oof_cat,oof_score_cat =train_catbo(X.drop(drop_fea,axis = 1), y,test_data.drop(drop_fea,axis = 1),params_cat)

params_xgb = {"objective":"multi:softprob",'learning_rate': 0.01,
          'num_class' :3, 'max_depth': 16}#, 'subsample': 0.9,

y_predicted_xgb,models_xgb,y_oof_xgb,oof_score_xgb=train_xgb(X.drop(drop_fea,axis = 1),
                                                             y.values,test_data.drop(drop_fea,axis = 1),params_xgb,num_iter=1500,es = 100,ve = 100)
def baseline_model(inp):
    # Create model here
    model = Sequential()
    model.add(Dense(100, input_dim = inp, activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dense(32, activation = 'relu'))
#     model.add(Dropout(0.3))
    model.add(Dense(20, activation = 'relu'))
    
    model.add(Dense(3, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    return model


seedEverything(42)
y_predicted_keras,models_keras,y_oof_keras,oof_score_keras = train_keras(X,
                                                                         train,test_data)
lr = LogisticRegression()
lr.fit(np.hstack([y_oof_keras,y_oof_xgb,y_oof_cat,y_oof_lgb]),y)
preds = lr.predict_proba(np.hstack([y_predicted_keras,y_predicted_xgb,y_predicted_cat,y_predicted_lgb]))

sub = preds_to_sub(test,le,preds,save = True)


# %% [code]
