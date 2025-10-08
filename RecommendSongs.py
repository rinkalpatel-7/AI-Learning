import dill
import csv
import pandas as pd
import numpy as np
train_df = pd.read_csv('train.csv')
# train_df.head()
all_songs = list(train_df['song_id'].unique())


def get_user_songs(uid):
    user_df = train_df[train_df['user_id'] == uid]
    user_songs = user_df['song_id'].unique()
    return list(user_songs)
# get_user_songs(train_df['user_id'][0])


def create_conc_matrix(usongs, asongs):
    usl = []
    for i in range(0, len(usongs)):
        usl.append(usongs[i])
    conc_matrix = np.matrix(np.zeros(shape=(len(usongs), len(asongs))), float)
    for i in range(0, len(asongs)):
        song_i = train_df[train_df['song_id'] == asongs[i]]
        user_i = set(song_i['user_id'].unique())
        for j in range(0, len(usongs)):
            user_j = usl[j]
            user_interaction = user_i.intersection(user_j)
            if len(user_interaction) != 0:
                user_union = user_i.union(user_j)
                conc_matrix[j, i] = float(len(user_interaction)/float(len(user_union)))
            else:
                conc_matrix[j, i] = 0
    return conc_matrix


def get_top_rec(uid, conc_mat, usongs, asongs):
    user_sim_score = conc_mat.sum(axis=0)/float(conc_mat.shape[0])
    user_sim_score = np.array(user_sim_score)[0]
    sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_score))), reverse=True)
    cols = ['user', 'song', 'score', 'rank']
    df = pd.DataFrame(columns=cols)
    rank = 1
    for i in range(0, len(sort_index)):
        if ~np.isnan(sort_index[i][0]) and asongs[sort_index[i][1]] not in usongs and rank<=10:
            df.loc[len(df)] = [uid, asongs[sort_index[i][1]], sort_index[i][0], rank]
            rank = rank + 1
    if df.shape[0] == 0:
        return -1
    else:
        return df


def rec_songs(uid):
    user_songs = get_user_songs(uid)
    matrix = create_conc_matrix(user_songs, all_songs)
    rec_df = get_top_rec(uid, matrix, user_songs, all_songs)
    return rec_df

# user_songs = get_user_songs(train_df['user_id'][0])
# mat = create_conc_matrix(user_songs,all_songs)


rec_u1 = rec_songs(train_df['user_id'][10])
rec_u1

# us = train_df[train_df['user_id']==train_df['user_id'][0]]
# us

