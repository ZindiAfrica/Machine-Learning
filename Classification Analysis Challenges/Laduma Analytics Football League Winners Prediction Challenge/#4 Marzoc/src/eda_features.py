# In[]
from functools import lru_cache
import streamlit as st
from collections import Counter
from tqdm import tqdm
import pandas as pd
from main import predict_number_of_goals
import matplotlib.pyplot as plt
from load_data import read_train, read_train_game_stats


def eda_goals_scored(train, train_game_stats):

    game_id = st.selectbox(
        'Game_ID',
        options=train_game_stats["Game_ID"].unique().tolist())

    game = train_game_stats[train_game_stats["Game_ID"] == game_id]

    game_info = train[train["Game_ID"] == game_id]

    goal_intervalls, goal_predicted_teams = predict_number_of_goals(
        game, verbose=True)

    teams = game["Team"].unique().tolist()

    goal_number = st.selectbox(
        'Select the interval that was predicted to contain a goal:',
        options=list(range(0, len(goal_intervalls))))
    intervall = goal_intervalls[goal_number]

    fig, ax = plt.subplots()
    intervall[intervall["Team"] == teams[0]]["X"].rolling(
        1).median().plot(label=teams[0])
    intervall[intervall["Team"] == teams[1]]["X"].rolling(
        1).median().plot(label=teams[1])
    plt.legend()
    plt.xlabel("observation")
    plt.ylabel("X-Position")
    st.pyplot(fig)

    team_scored = intervall[intervall["Goals_scored"] == 1]["Team"].iloc[0]
    st.markdown(f"Team scored: {team_scored}")


@lru_cache
def get_number_of_games_where_simple_method_fails():
    train_game_stats = read_train_game_stats()
    difficult_games = []
    for id, game in train_game_stats.groupby("Game_ID"):
        try:
            goal_intervalls, goal_predicted_teams = predict_number_of_goals(
                game)
        except:
            difficult_games.append(id)

    return len(difficult_games) / train_game_stats["Game_ID"].nunique()


def predict_winning_team(game):
    goal_intervalls, goal_predicted_teams = predict_number_of_goals(game)
    dict = Counter(goal_predicted_teams)
    dict2 = {}
    for k, v in dict.items():
        dict2[k] = [v]

    df_pred = pd.DataFrame.from_dict(dict2)

    winning_team_pred = "Draw"
    teams = list(set(goal_predicted_teams))
    if len(teams) > 0:
        first_team = teams[0]
        winning_team_pred = first_team

        if len(teams) > 1:
            second_team = teams[1]
            if dict[first_team] > dict[second_team]:
                winning_team_pred = first_team
            elif dict[first_team] < dict[second_team]:
                winning_team_pred = second_team
            elif dict[first_team] == dict[second_team]:
                winning_team_pred = "Draw"
    return winning_team_pred


@lru_cache
def evaluate_simple_non_ml_predictor():

    num_correct_pred = 0

    for id, game in tqdm(train_game_stats.groupby("Game_ID")):
        try:
            goals_df_gt = game[game["Goals_scored"]
                               == 1.0]["Team"].value_counts()
            winning_team_gt = "Draw"
            if len(goals_df_gt) > 0:
                winning_team_gt = goals_df_gt.index[goals_df_gt.argmax()]

                if len(goals_df_gt) > 1:
                    if goals_df_gt.iloc[0] == goals_df_gt.iloc[1]:
                        winning_team_gt = "Draw"

            winning_team_pred = predict_winning_team(game)

            if False:
                print("_"*20)
                print(id)
                print(f"winner: {winning_team_gt}")
                print(f"winner_pred: = {winning_team_pred}")
                print("_"*20)

            if winning_team_gt == winning_team_pred:
                num_correct_pred += 1

        except:
            print(f"something went wrong with game {id}")

    st.markdown(
        f"correctly predicted game outcomes: {num_correct_pred/train_game_stats['Game_ID'].nunique()}")


def predicting_number_of_goals():
    eda_goals_scored(train, train_game_stats)


train = read_train()
train_game_stats = read_train_game_stats()


def main_page():
    st.markdown("# Predicting Outcomes of Footbal-matches")
    st.markdown(
        """
        ```python
        game = game.sort_values(by=["Start_minutes", "End_minutes"])
        game["End_minutes_max_diff_current"] = game["End_minutes"].cummax() - game["End_minutes"]
        ```
        In Most cases this yields a feature that can be used to predict accurately if a goal has been scored.
        """
    )

    eda_goals_scored(train, train_game_stats)

    difficult_games()

    evaluate_non_ml_approach()


def difficult_games():
    ratio = get_number_of_games_where_simple_method_fails()
    st.markdown(
        f"can accurately predict if a goal was scored in {1.0-ratio} of the cases.")


def evaluate_non_ml_approach():
    evaluate_simple_non_ml_predictor()


main_page()

# page_names_to_funcs = {
#     "Main Page": main_page,
#     "Predict Number of Goals": predicting_number_of_goals,
#     "Difficult Games": difficult_games,
#     "Evaluation of Non-ML-Approach": evaluate_non_ml_approach

# }

# selected_page = st.sidebar.selectbox(
#     "Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()
