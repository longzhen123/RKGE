import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import argparse
from src.load_base import load_ratings, data_split, get_records, get_rec


def construct_kg(data_dir, train_records, user_set):

    kg = nx.DiGraph()
    triple_np = pd.read_csv(data_dir + 'kg.txt', delimiter='\t', header=None).values
    head_set = set(triple_np[:, 0])
    tail_set = set(triple_np[:, 1])
    entity_list = list(head_set.union(tail_set)) + list(user_set)
    for triple in triple_np:
        head = entity_list.index(triple[0])
        tail = entity_list.index(triple[1])
        relation = triple[2]

        kg.add_edge(head, tail, relation=relation)

    n_relation = len(set(triple_np[:, 2]))

    for user in train_records:
        for item in train_records[user]:
            kg.add_edge(entity_list.index(user), entity_list.index(item), relation=n_relation)
            kg.add_edge(entity_list.index(item), entity_list.index(user), relation=n_relation + 1)
    return kg, entity_list


def get_relation_dict(kg):

    relation_dict = dict()
    for edge in tqdm(kg.edges):

        relation_dict[(edge[0], edge[1])] = kg[edge[0]][edge[1]]['relation']

    return relation_dict


def get_paths(args):
    np.random.seed(555)
    data_dir = './data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    train_set, eval_set, test_set = data_split(ratings_np, args.ratio)
    item_set = set(ratings_np[:, 1])
    user_set = set(ratings_np[:, 0])
    train_records = get_records(train_set)
    eval_records = get_records(eval_set)
    test_records = get_records(test_set)
    rec = get_rec(train_records, eval_records, test_records, item_set)
    kg, entity_list = construct_kg(data_dir, train_records, user_set)
    relation_dict = get_relation_dict(kg)

    path_dict = dict()
    new_train_set = []

    for pair in tqdm(train_set):
        user = entity_list.index(pair[0])
        item = entity_list.index(pair[1])
        new_train_set.append([user, item, pair[2]])

        paths = [path for path in list(nx.all_simple_paths(kg, user, item, cutoff=args.path_len)) if (len(path) == args.path_len + 1)]

        if len(paths) > 50:
            indices = np.random.choice(len(paths), 50, replace=False)
            paths = [paths[i] for i in indices]

        path_dict[(user, item)] = paths

    new_eval_set = []

    for pair in tqdm(eval_set):
        user = entity_list.index(pair[0])
        item = entity_list.index(pair[1])
        new_eval_set.append([user, item, pair[2]])

        paths = [path for path in list(nx.all_simple_paths(kg, user, item, cutoff=args.path_len)) if
                 (len(path) == args.path_len + 1)]

        if len(paths) > 50:
            indices = np.random.choice(len(paths), 50, replace=False)
            paths = [paths[i] for i in indices]

        path_dict[(user, item)] = paths

    new_test_set = []

    for pair in tqdm(test_set):
        user = entity_list.index(pair[0])
        item = entity_list.index(pair[1])
        new_test_set.append([user, item, pair[2]])

        paths = [path for path in list(nx.all_simple_paths(kg, user, item, cutoff=args.path_len)) if
                 (len(path) == args.path_len + 1)]

        if len(paths) > 50:
            indices = np.random.choice(len(paths), 50, replace=False)
            paths = [paths[i] for i in indices]

        path_dict[(user, item)] = paths

    new_rec = dict()
    for user in tqdm(rec):
        new_rec[entity_list.index(user)] = [entity_list.index(i) for i in rec[user]]
        new_user = entity_list.index(user)
        for item in new_rec[new_user]:

            paths = [path for path in list(nx.all_simple_paths(kg, new_user, item, cutoff=args.path_len)) if (len(path) == args.path_len + 1)]

            if len(paths) > 50:
                indices = np.random.choice(len(paths), 50, replace=False)
                paths = [paths[i] for i in indices]

            path_dict[(new_user, item)] = paths

    np.save(data_dir+str(args.ratio)+'_'+str(args.path_len)+'_path_dict.npy', path_dict)
    np.save(data_dir+str(args.ratio)+'_relation_dict.npy', relation_dict)
    np.save(data_dir+'_entity_list.npy', entity_list)
    np.save(data_dir + str(args.ratio)+'_train_set.npy', new_train_set)
    np.save(data_dir + str(args.ratio) + '_eval_set.npy', new_eval_set)
    np.save(data_dir + str(args.ratio)+'_test_set.npy', new_test_set)
    np.save(data_dir + str(args.ratio)+'_rec.npy', new_rec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    parser.add_argument('--path_len', type=int, default=3, help='The length of paths')
    parser.add_argument('--ratio', type=float, default=0.8, help='The ratio of training set')
    args = parser.parse_args()

    get_paths(args)