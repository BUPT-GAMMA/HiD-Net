import argparse
import torch
import numpy as np
import warnings
import os

from load_data import load_data
from model import ReactionNet
from pandas import Series,DataFrame
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from utils import read_config
# from memory_profiler import profile
from earlystopping import EarlyStopping
from earlystopping import stopping_args
import time
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', type=str, default='False', help='if tune')
    parser.add_argument('--configfile', type=str, default='1111', help='configfile')
    parser.add_argument('--predictfile', type=str, default='1111', help='predictfile')
    parser.add_argument('--times', type=int, default=3, help='config times')
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--repeat', type=int, default=5, help='repeat time')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=64, help='hidden size')
    parser.add_argument('--head1', type=int, default=1, help='gat head1')
    parser.add_argument('--head2', type=int, default=1, help='gat head2')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--drop', type=str, default='False', help='whether to dropout or not')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu'])
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'sbm'])
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train the base model')
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--alpha', type=float, default=0, help='tolerance to stop EM algorithm')
    parser.add_argument('--beta', type=float, default=1, help='tolerance to stop EM algorithm')
    parser.add_argument('--gamma', type=float, default=0, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma1', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma2', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--calg', type=str, default='cal_gradient_2', help='calculate gradient')  # TODO
    parser.add_argument('--gpu',  default='9', type=int, help='-1 means cpu')
    parser.add_argument('--earlystop', type=bool, default=False, help='if tune')
    parser.add_argument('--patience', type=int, default=50, help='if tune')
    parser.add_argument('--reg_lambda', type=float, default=0, help='if tune')
    parser.add_argument('--print_interval', type=int, default=100, help='if tune')
    parser.add_argument('--variable', type=str, default='none', help='if tune')
    parser.add_argument('--value', type=float, default=0, help='if tune')
    parser.add_argument('--mn', type=str, default='gdgc', help='if tune')
    args = parser.parse_args()
    if args.tune == 'True':
        args = read_config(args)

    # TODO
    if args.variable == 'alpha':
        args.alpha = args.value
    elif args.variable == 'beta':
        args.beta = args.value
    elif args.variable == 'gamma':
        args.gamma = args.value
    elif args.variable == 'k':
        args.k = int(args.value)
    else:
        pass

    print(args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dataset = load_data("./data/hetero", args.dataset)

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        data = dataset.data
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        i = args.split
        dataset.process()
        data = dataset.data
        train_mask = data['train_mask'].T[i]
        val_mask = data['val_mask'].T[i]
        test_mask = data['test_mask'].T[i]

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    # criterion = F.nll_loss()
    for seed in range(args.repeat):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model = ReactionNet(args, dataset.num_features, dataset.num_classes).to(device)
            #
            # model = ReactionNetn(args, dataset.num_features, dataset.num_classes).to(device)
        # TODO
        reg_lambda = torch.tensor(args.reg_lambda, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Define optimizer.

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []

        early_stopping = EarlyStopping(model, **stopping_args)
        start_time = time.time()
        last_time = start_time
        for epoch in range(early_stopping.max_epochs):
            # train(data.to(device))
            # train
            data.to(device)
            model.train()

            optimizer.zero_grad()  # Clear gradients.
            train_out = model(data.x, data.edge_index)  # Perform a single forward pass.
            # train_loss = criterion(train_out[train_mask], data.y[train_mask])
            train_loss = F.nll_loss(train_out[train_mask], data.y[train_mask])

            # TODO
            l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
            train_loss = train_loss + args.reg_lambda / 2 * l2_reg

            # Compute the loss solely based on the training nodes.
            train_loss.backward()  # Derive gradients.

            optimizer.step()

            train_preds = torch.argmax(train_out, dim=1)
            #train
            train_acc = torch.sum(train_preds[train_mask] == data.y[train_mask]).float() / data.y[train_mask].shape[0]

            # val
            model.eval()
            val_out = model(data.x, data.edge_index) # re calculate
            # val_loss = criterion(val_out[val_mask], data.y[val_mask])
            val_loss = F.nll_loss(val_out[val_mask], data.y[val_mask])
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = torch.sum(val_preds[val_mask] == data.y[val_mask]).float() / data.y[val_mask].shape[0]
            val_f1_macro = f1_score(data.y[val_mask].cpu(), val_preds[val_mask].cpu(), average='macro')
            val_f1_micro = f1_score(data.y[val_mask].cpu(), val_preds[val_mask].cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test

            test_acc = torch.sum(val_preds[test_mask] == data.y[test_mask]).float() / data.y[test_mask].shape[0]
            test_f1_macro = f1_score(data.y[test_mask].cpu(), val_preds[test_mask].cpu(), average='macro')
            test_f1_micro = f1_score(data.y[test_mask].cpu(), val_preds[test_mask].cpu(), average='micro')
            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(val_out[test_mask])

            print(f"Epoch {epoch}: "
                  f"Train loss = {train_loss:.2f}, "
                  f"train acc = {train_acc * 100:.2f}, "
                  f"val loss = {val_loss:.2f}, "
                  f"val acc = {val_acc * 100:.2f} "
                  f"test acc = {test_acc * 100:.2f} "
                  )


            if len(early_stopping.stop_vars) > 0:
                stop_vars = [val_acc.item(), val_loss.item()]
                if early_stopping.check(stop_vars, epoch):
                    break



        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

            # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=data.y[test_mask].detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy(),
                                                multi_class='ovr'))
        print("\tMacro-F1: {:.4f}  Micro-F1: {:.4f} acc {:.4f}"
              .format(test_macro_f1s[max_iter],
                      test_micro_f1s[max_iter],
                      test_accs[max_iter],
                      )
              )
        if args.dataset == 'cora' and test_accs[max_iter] < 0.835:
            break
        elif args.dataset == 'citeseer' and test_accs[max_iter] < 0.715:
            break
        elif args.dataset == 'pubmed' and test_accs[max_iter] < 0.80:
            break
        elif args.dataset == 'actor' and test_accs[max_iter] < 0.34:
            break
        elif args.dataset == 'chameleon' and test_accs[max_iter] < 0.59:
            break
        elif args.dataset == 'squirrel' and test_accs[max_iter] < 0.41:
            break
        else:
            pass

    print("\t[Classification]Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
        .format(
            np.mean(macro_f1s),
            np.std(macro_f1s),
            np.mean(micro_f1s),
            np.std(micro_f1s),
            np.mean(auc_score_list),
            )
        )
            # train_acc, val_acc, tmp_test_acc = test(data)
            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     test_acc = tmp_test_acc
            # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
            #     f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

    if args.tune == 'True':
        result = {
            'model': ['gdgc'],
            'config_file': [args.configfile],
            'dataset': [args.dataset],
            'calg': [args.calg],
            'times': [args.times],

            'alpha': [args.alpha],
            'beta': [args.beta],
            'gamma': [args.gamma],

            'k': [args.k],
            'hidden': [args.hidden],

            'drop': [args.drop],
            'dropout': [args.dropout],
            'lr': [args.lr],
            'weight_decay': [args.weight_decay],

            'sigma1': [args.sigma1],
            'sigma2': [args.sigma2],

            'reg_lambda': [args.reg_lambda],

            'Macro-F1_mean': [np.mean(macro_f1s)],
            'Macro-F1_var': [np.std(macro_f1s)],
            'Micro-F1_mean': [np.mean(micro_f1s)],
            'Micro-F1_var': [np.std(micro_f1s)],
            'auc_mean': [np.mean(auc_score_list)],
            'auc_var': [np.std(auc_score_list)],
            'acc_mean': [np.mean(accs)],
            'acc_var': [np.std(accs)]

        }

        df = DataFrame(result)
        print(df)
        path = 'prediction/excel/{}'.format(args.predictfile)
        if not os.path.exists(path):
            os.makedirs(path)
        if args.variable == 'none':
            df.to_csv('{}/{}_{}_{}.csv'.format(path, args.dataset, args.calg, args.times))
        elif args.variable == 'k':
            df.to_csv('{}/gdgc_{}_{}_{}.csv'.format(path, args.dataset, args.times, args.k))
        else:
            df.to_csv('{}/{}_{}_{}_{}_{}.csv'.format(path, args.dataset, args.calg, args.times, args.variable, args.value))


if __name__=='__main__':
    main()
    os._exit(0)




