from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.base import clone as clone_model

from load_data import DATA, SUBMISSIONS


def preprocess(train, test):
    train['Train'] = 1
    test['Train'] = 0
    home_teams = train['Home Team'].unique()
    all_data = pd.concat([train, test])
    for team in home_teams:
        all_data['home_'+team] = 0
        all_data.loc[all_data['Home Team'] == team, 'home_'+team] = 1
    away_teams = train['Away Team'].unique()
    for team in away_teams:
        all_data['away_'+team] = 0
        all_data.loc[all_data['Away Team'] == team, 'away_'+team] = 1
    train_cols = all_data.columns.difference(
        ['Date', 'Season', 'Match_ID', 'Game_ID', 'Score', 'Train', 'Home Team', 'Away Team'])
    all_data.fillna(all_data.mean(), inplace=True)
    train = all_data[all_data["Train"] == 1]
    test = all_data[all_data["Train"] == 0]
    return train[train_cols], test[train_cols], train['Score'], test['Score']


def make_mkmworld_predictions(submission_name="", model_type="lr"):
    train = pd.read_csv(DATA / "Train.csv", parse_dates=['Date'])
    test = pd.read_csv(DATA / "Test.csv", parse_dates=['Date'])

    le = LabelEncoder()
    train["Score"] = le.fit_transform(train["Score"])
    score_mapping = dict(zip(le.classes_, range(len(le.classes_))))

    train.sort_values(by=['Date'], inplace=True)
    trai, val = train_test_split(train, test_size=0.2, shuffle=False)
    pro_train_X, pro_test_X, pro_train_y, pro_test_y = preprocess(trai, val)

    X, test_data, y, _ = preprocess(train, test)

    if model_type == "lr":
        lr = LogisticRegression(C=0.1, max_iter=1000)
        lr.fit(pro_train_X, pro_train_y)
    else:
        lr = CatBoostClassifier(max_depth=12, iterations=1000)
        lr.fit(pro_train_X, pro_train_y, eval_set=(pro_test_X, pro_test_y))

    y_pred = lr.predict_proba(pro_test_X)

    lr2 = clone_model(lr)

    lr2.fit(X, y)
    test_predictions = lr.predict_proba(test_data)
    Test = test.copy()
    cols = le.inverse_transform([*range(3)])
    Test[cols] = test_predictions

    submit = Test[["Game_ID", 'Away win', 'Draw', 'Home Win']]
    submit.drop_duplicates(subset=["Game_ID"], inplace=True)
    submit = submit.reset_index(drop=True)

    if submission_name != "":
        submit.to_csv(SUBMISSIONS / submission_name, index=False)

    return submit, log_loss(pro_test_y, y_pred)


if __name__ == "__main__":
    make_mkmworld_predictions()
