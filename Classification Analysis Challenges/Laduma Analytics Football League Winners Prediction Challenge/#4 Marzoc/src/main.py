import argparse
from load_data import ROOT, SUBMISSIONS
from pathlib import Path
from simple_model_mkmworld import make_mkmworld_predictions
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import read_train_game_stats, read_train, read_test, read_test_game_stats, read_sample_submission

import streamlit as st

SEED = 42
np.random.seed(SEED)

RELEVANT_COLUMNS = ["X", "Y", "Player_ID",
                    "Start_minutes", "End_minutes", "Team"]

RELEVANT_COLUMNS_TARGET = ["Goals_scored"]

RELEVANT_COLUMNS_TRAIN = RELEVANT_COLUMNS + RELEVANT_COLUMNS_TARGET


def predict_number_of_goals(game, verbose=False):

    if "Goals_scored" in game.columns:
        game = game[RELEVANT_COLUMNS_TRAIN]
    else:
        game = game[RELEVANT_COLUMNS]

    game = game.sort_values(by=["Start_minutes", "End_minutes"])

    if "Goals_scored" in game.columns:
        num_goals = int(game["Goals_scored"].sum())
        if verbose:
            print("num_goals", num_goals)
            st.markdown(f"actual number of goals in game: {num_goals}")

    game["End_minutes_max_diff_current"] = game["End_minutes"].cummax() - \
        game["End_minutes"]

    if verbose:
        fig, ax = plt.subplots()
        game["End_minutes_max_diff_current"].plot()
        plt.title("End_minutes_max_diff_current")
        st.pyplot(fig)

    game = game.reset_index(drop=True)
    indices = game[game["End_minutes_max_diff_current"] > 0].index.tolist()

    goal_indices_all = game.iloc[indices].copy()
    goal_indices_all["End_minutes_prev_diff"] = goal_indices_all["End_minutes"] - \
        goal_indices_all["End_minutes"].shift(1)

    if verbose:
        goal_indices_all["End_minutes_prev_diff"].plot()
        plt.show()

    goal_indices_all["New_goal_interval"] = (
        goal_indices_all["End_minutes_prev_diff"] > 1).astype(int).copy()

    goal_indices_all["Goal_num"] = goal_indices_all["New_goal_interval"].cumsum()

    goal_intervalls = []
    goal_predicted_teams = []
    for goal_num, goal_interval in goal_indices_all.groupby("Goal_num"):

        indices = goal_interval[goal_interval["End_minutes_max_diff_current"]
                                > 0].index.tolist()

        dt = 2
        indices = list(range(min(indices)-dt, max(indices)+dt-1))
        intervall = game.iloc[indices]

        if "Goals_scored" in game.columns:
            goals_in_interval = int(intervall["Goals_scored"].sum())
            # assert goals_in_interval == 1, f"{goals_in_interval}"

        goal_intervalls.append(intervall)

        # predict the scoring team
        med_x = intervall.groupby("Team")["X"].mean()
        team_pred = med_x.index[med_x.argmax()]
        goal_predicted_teams.append(team_pred)

    num_predicted_goals = goal_indices_all["Goal_num"].nunique()

    if "Goals_scored" in game.columns:
        assert num_predicted_goals == num_goals, f"{num_predicted_goals} != {num_goals}"

    if verbose:
        print(num_predicted_goals)

    return goal_intervalls, goal_predicted_teams


def get_X_y(game, game_id, info, X_li, y_li=None, verbose=False):
    goal_intervalls, goal_predicted_teams = predict_number_of_goals(
        game, verbose=verbose)

    team_names = info[info["Game_ID"] == game_id][[
        "Home Team", "Away Team"]].iloc[0].to_dict()

    team_names_inv = {}
    for k, v in team_names.items():
        team_names_inv[v] = k

    for intervall in goal_intervalls:
        intervall["Team"] = intervall["Team"].apply(
            lambda x: team_names_inv[x])

        cols = ["X", "Y", "Team"]
        X_g = intervall[cols]

        max_values = 40
        X_g = X_g.iloc[:max_values]

        X_ = X_g.stack().to_frame().T
        X_.columns = [
            c + f"_{i}" for i in range(0, len(X_g)) for c in cols]

        if len(X_g) < max_values:
            max_na_cols = [
                c + f"_{i}" for i in range(len(X_g), max_values) for c in cols]
            X_fill_na = pd.DataFrame(columns=max_na_cols)

            X_ = pd.concat([X_, X_fill_na], axis=0)

        if y_li is not None:
            team_scored = intervall[intervall["Goals_scored"]
                                    == 1.0]["Team"].iloc[0]
            y_g = team_scored
            y_li.append(pd.DataFrame({"Winner": [y_g]}))

        X_li.append(X_.reset_index(drop=True))

    return X_li, y_li


def get_X_y_all_games(train, train_game_stats):
    X_li = []
    y_li = []

    for game_id, game in tqdm(train_game_stats.groupby("Game_ID")):

        try:
            X_li, y_li = get_X_y(game, game_id, train, X_li, y_li)
        except:
            print(game_id)

    X = pd.concat(X_li).reset_index(drop=True)
    y = pd.concat(y_li).reset_index(drop=True)

    team_cols = [c for c in X.columns if "Team_" in c]
    non_team_cols = list(set(X.columns) - set(team_cols))

    return X, y, team_cols, non_team_cols


def blend_submissions(submission_1, submission_2, w=(0.5, 0.5)):
    final_submission_blend = submission_1.copy()
    cols = ["Away win", "Draw", "Home Win"]

    final_submission_blend[cols] = submission_1[cols] * \
        w[0] + submission_2[cols]*w[1]

    final_submission_blend[cols] = final_submission_blend[cols].div(
        final_submission_blend[cols].sum(axis=1), axis=0)

    assert abs(final_submission_blend[cols].sum().sum(
    ) - len(submission_1)) < 0.01, "normalization failed"
    return final_submission_blend


def monte_carlo_simulate_wins(arr_stacked, repeats=100000):

    team_1_win = 0
    team_2_win = 0
    draw = 0

    for t in range(0, repeats):
        goals_team_1 = 0
        goals_total = len(arr_stacked)
        for i in range(0, len(arr_stacked)):
            p = arr_stacked[i, 0]
            res = np.random.binomial(n=1, p=p)
            goals_team_1 += res

        goals_team_2 = goals_total - goals_team_1
        if goals_team_1 == goals_team_2:
            draw += 1
        elif goals_team_1 > goals_team_2:
            team_1_win += 1
        elif goals_team_1 < goals_team_2:
            team_2_win += 1

    df = pd.DataFrame({"Home Win": [team_1_win], "Draw": [
                      draw], "Away win": [team_2_win]})
    return df.div(df.sum(axis=1), axis=0)


def evaluate_catboost_model(X, y, team_cols, non_team_cols, catboost_iterations):
    kf = KFold(n_splits=5)

    y_preds = []
    y_vals = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        ss_cat = SimpleImputer(
            missing_values=np.nan, fill_value="-100", strategy="most_frequent").fit(X_train[team_cols])
        ss_num = SimpleImputer(
            missing_values=np.nan, fill_value=-100).fit(X_train[non_team_cols])

        X_train[team_cols] = ss_cat.transform(X_train[team_cols])
        X_train[non_team_cols] = ss_num.transform(X_train[non_team_cols])

        X_val[team_cols] = ss_cat.transform(X_val[team_cols])
        X_val[non_team_cols] = ss_num.transform(X_val[non_team_cols])

        pipe = CatBoostClassifier(
            max_depth=9, iterations=catboost_iterations, cat_features=team_cols)
        pipe.fit(X_train, y_train, eval_set=(X_val, y_val))

        pred = pipe.predict(X_val)

        y_preds.extend(pred.tolist())
        y_vals.extend(y_val["Winner"].tolist())

    print(confusion_matrix(y_vals, y_preds))
    print(accuracy_score(y_vals, y_preds))


def train_catboost_model(X, y, team_cols, non_team_cols, catboost_iterations):
    X_ = X.copy()
    ss_cat = SimpleImputer(
        missing_values=np.nan, fill_value="-100", strategy="most_frequent").fit(X_[team_cols])
    ss_num = SimpleImputer(
        missing_values=np.nan, fill_value=-100).fit(X_[non_team_cols])

    X_[team_cols] = ss_cat.transform(X_[team_cols])
    X_[non_team_cols] = ss_num.transform(X_[non_team_cols])

    pipe = CatBoostClassifier(
        max_depth=9, iterations=catboost_iterations, cat_features=team_cols)
    pipe.fit(X_, y)

    return pipe, ss_cat, ss_num


def predict_game_winners(test, test_game_stats, team_cols, non_team_cols, pipe, ss_cat, ss_num, monte_carlo_iterations):
    win_pred_df_li = []
    for game_id, game in tqdm(test_game_stats.groupby("Game_ID")):
        X_test_li, _ = get_X_y(game, game_id, test, [], verbose=False)

        goal_predicted_probs = []
        for X_ in X_test_li:
            X_[team_cols] = ss_cat.transform(X_[team_cols])
            X_[non_team_cols] = ss_num.transform(X_[non_team_cols])
            pred = pipe.predict_proba(X_)
            goal_predicted_probs.append(pred)

        win_pred_df = pd.DataFrame(
            {"Home Win": [0.0], "Draw": [1.0], "Away win": [0.0]})

        if len(goal_predicted_probs) > 0:
            arr_stacked = np.vstack(goal_predicted_probs)
            win_pred_df = monte_carlo_simulate_wins(
                arr_stacked, monte_carlo_iterations)

        win_pred_df["Game_ID"] = game_id
        win_pred_df_li.append(win_pred_df)

    win_pred_df_final = pd.concat(win_pred_df_li, axis=0)
    print(win_pred_df_final.shape)

    ss = read_sample_submission()
    print(ss.shape)

    final_submission = ss[["Game_ID"]].reset_index(drop=True).merge(
        win_pred_df_final.reset_index(drop=True), on="Game_ID")

    return final_submission


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model to predict outcomes of footbal-matches.')
    parser.add_argument('--catboost_iterations', type=int, default=1000,
                        help='number of iterations to train the catboost-model.')
    parser.add_argument('--monte_carlo_iterations', type=int, default=100000,
                        help='number of repeats during monte-carlo-simulation.')
    parser.add_argument('--evaluate', type=bool, default=False,
                        help='specifies if the model is evaluated.')

    return parser.parse_args()


def main():
    args = parse_args()

    train_game_stats = read_train_game_stats()
    train = read_train()

    test_game_stats = read_test_game_stats()
    test = read_test()

    X, y, team_cols, non_team_cols = get_X_y_all_games(train, train_game_stats)

    if args.evaluate:
        evaluate_catboost_model(
            X, y, team_cols, non_team_cols, args.catboost_iterations)
    else:
        pipe, ss_cat, ss_num = train_catboost_model(
            X, y, team_cols, non_team_cols, args.catboost_iterations)

        submission = predict_game_winners(
            test, test_game_stats, team_cols, non_team_cols, pipe, ss_cat, ss_num, args.monte_carlo_iterations)

        mkmworld_predictions, _ = make_mkmworld_predictions()

        WEIGHTS_BLEND = (0.95, 0.05)
        submission_blend = blend_submissions(
            submission, mkmworld_predictions, w=WEIGHTS_BLEND)

        submission_id = f"simple_2_009_blend_{WEIGHTS_BLEND[0]}_{WEIGHTS_BLEND[1]}"
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        submission_file = SUBMISSIONS / Path(submission_id + ".csv")
        submission_blend.to_csv(submission_file, index=False)


if __name__ == "__main__":
    main()
