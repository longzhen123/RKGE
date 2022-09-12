import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.evaluate import get_all_metrics
from src.load_base import get_records, load_kg


class RKGE(nn.Module):

    def __init__(self, n_entity, dim):

        super(RKGE, self).__init__()
        self.dim = dim
        self.entity_embedding_matrix = nn.Parameter(t.randn(n_entity, dim))
        self.rnn = nn.RNN(dim, dim, num_layers=1, bidirectional=True)
        self.M = nn.Linear(2*dim, 1)
        self.W = nn.Linear(dim, dim, bias=False)
        self.H = nn.Linear(dim, dim)
        self.Wr = nn.Linear(dim, 1)

    def forward(self, paths_list):

        path_embeddings = self.get_path_embedding(paths_list)

        # (len, -1, 2 * dim)
        hidden_states = self.rnn(path_embeddings)[0]

        for i in range(4):
            attention = t.sigmoid(self.M(hidden_states[0]))

            if i == 0:
                candidate_hidden_states = t.sigmoid(self.H(path_embeddings[i]))
                next_hidden_states = attention * candidate_hidden_states
            else:
                candidate_hidden_states = t.sigmoid(self.W(next_hidden_states) + self.H(path_embeddings[i]))
                next_hidden_states = (1 - attention) * next_hidden_states + attention * candidate_hidden_states
        aggregation_hidden_states = next_hidden_states.reshape(len(paths_list), -1, self.dim).mean(dim=1)
        predicts = t.sigmoid(self.Wr(aggregation_hidden_states)).reshape(-1)

        return predicts

    def get_path_embedding(self, paths_list):
        path_embedding_list = []
        zeros = t.zeros(4, 5, self.dim)
        if t.cuda.is_available():
            zeros = zeros.to(self.entity_embedding_matrix.data.device)
        for paths in paths_list:
            entity_embedding_list = []
            for path in paths:
                # (len, 1, dim)
                entity_embedding_list.append(self.entity_embedding_matrix[path].reshape(-1, 1, self.dim))

            # (len, n_path, dim)
            if len(paths) == 0:
                path_embedding_list.append(zeros)
                continue
            path_embedding_list.append(t.cat(entity_embedding_list, dim=1))

        return t.cat(path_embedding_list, dim=1)


def get_scores(model, rec, paths_dict, p):
    scores = {}
    model.eval()
    for user in (rec):

        pairs = [(user, item, -1) for item in rec[user]]
        paths_list = get_data(pairs, paths_dict, p)[0]

        predict_list = model(paths_list).cpu().detach().numpy().tolist()

        item_scores = dict()
        i = 0
        for item in rec[user]:

            item_scores[item] = predict_list[i]
            i += 1

        sorted_item_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        scores[user] = [i[0] for i in sorted_item_scores]
    model.train()
    return scores


def eval_ctr(model, pairs, paths_dict, p, batch_size):

    model.eval()
    pred_label = []
    paths_list = get_data(pairs, paths_dict, p)[0]
    for i in range(0, len(pairs), batch_size):
        batch_label = model(paths_list[i: i+batch_size]).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np  = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_data(pairs, paths_dict, p):
    paths_list = []
    label_list = []
    for pair in pairs:
        if len(paths_dict[(pair[0], pair[1])]):
            paths = paths_dict[(pair[0], pair[1])]

            if len(paths) >= p:
                indices = np.random.choice(len(paths), p, replace=False)
            else:
                indices = np.random.choice(len(paths), p, replace=True)

            paths_list.append([paths[i] for i in indices])
        else:
            paths_list.append([])

        label_list.append(pair[2])
    return paths_list, label_list


def train(args, is_topk=False):
    np.random.seed(555)

    data_dir = './data/' + args.dataset + '/'
    train_set = np.load(data_dir + str(args.ratio) + '_train_set.npy').tolist()
    eval_set = np.load(data_dir + str(args.ratio) + '_eval_set.npy').tolist()
    test_set = np.load(data_dir + str(args.ratio) + '_test_set.npy').tolist()
    test_records = get_records(test_set)
    entity_list = np.load(data_dir + '_entity_list.npy').tolist()
    _, _, n_relation = load_kg(data_dir)
    n_entity = len(entity_list)
    rec = np.load(data_dir + str(args.ratio) + '_rec.npy', allow_pickle=True).item()
    paths_dict = np.load(data_dir + str(args.ratio) + '_3_path_dict.npy', allow_pickle=True).item()

    model = RKGE(n_entity, args.dim)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end=', ')
    print('p: %d' % args.p, end=', ')
    print('lr: %1.0e' % args.lr, end=', ')
    print('l2: %1.0e' % args.l2, end=', ')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []

    for epoch in range(args.epochs):
        loss_sum = 0
        start = time.clock()
        np.random.shuffle(train_set)
        paths, true_label = get_data(train_set, paths_dict, args.p)

        labels = t.tensor(true_label).float()
        if t.cuda.is_available():
            labels = labels.to(args.device)
        start_index = 0
        size = len(paths)
        model.train()
        while start_index < size:

            predicts = model(paths[start_index: start_index + args.batch_size])
            loss = criterion(predicts, labels[start_index: start_index + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        train_auc, train_acc = eval_ctr(model, train_set, paths_dict, args.p, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, paths_dict, args.p, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, paths_dict, args.p, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec, paths_dict, args.p)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]



