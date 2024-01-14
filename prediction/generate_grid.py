import yaml
import argparse
import os


hidden = [64]
dropout = [0.5, 0.6]
lr = [0.01]
weight_decay = [5e-3, 5e-4]
k = [1, 2, 4, 6, 8, 10]
alpha = [0.1, 0.2, 0.3]
gamma = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]
reg_lambda = [0.005]
drop = ['True', 'False']

# hidden = [64]
# dropout = [0.6]
# lr = [0.01]
# weight_decay = [5e-3]
# k = [10]
# alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# gamma = [-0.3]
# reg_lambda = [0.005]
# drop = ['True']

# hidden = [64]
# dropout = [0.5, 0.55]
# lr = [0.005, 0.006, 0.007, 0.008, 0.009]
# weight_decay = [0.05, 0.06, 0.07, 0.08]
# k = [10]
# alpha = [0.225, 0.22, 0.23, 0.235,]
# gamma = [-0.23, -0.25, -0.27, -0.29, -0.31]
# reg_lambda = [0.005,]
# drop = ['True']

def getIndex():
    dicts = {}

    count = 0

    for a in range(len(hidden)):
        dict = {}
        dict['hidden'] = a
        for b in range(len(dropout)):
            dict['dropout'] = b
            for c in range(len(lr)):
                dict['lr'] = c
                for d in range(len(weight_decay)):
                    dict['weight_decay'] = d
                    for z in range(len(k)):
                        dict['k'] = z
                        for e in range(len(alpha)):
                            dict['alpha'] = e
                            for x in range(len(drop)):
                                dict['drop'] = x
                                for y in range(len(gamma)):
                                    dict['gamma'] = y
                                    for z in range(len(reg_lambda)):
                                        dict['reg_lambda'] = z
                                        dicts[count] = dict.copy()
                                        count = count + 1

    return dicts


def makeDict(index):
    dict = {
        'hidden': hidden[index['hidden']],
        'drop': drop[index['drop']],
        'dropout': dropout[index['dropout']],
        'lr': lr[index['lr']],
        'weight_decay': weight_decay[index['weight_decay']],
        'k': k[index['k']],
        'alpha': alpha[index['alpha']],
        'beta': 1 - alpha[index['alpha']],
        'gamma': gamma[index['gamma']],
        'reg_lambda': reg_lambda[index['reg_lambda']]

    }
    return dict

def generate(configfile):
    datasets = ['cora', 'citeseer', 'pubmed', 'sbm', 'actor', 'chameleon', 'squirrel']

    dicts = {}

    indexes = getIndex()

    size = len(indexes)
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    if not os.path.exists('./prediction/config/{}'.format(configfile)):
        os.makedirs('./prediction/config/{}'.format(configfile))

    for count in range(size):
        index = indexes[count]
        for a in datasets:
            dict = makeDict(index)
            dicts[a] = dict
        aproject = {
                    'g2': dicts,
                    'g3': dicts,
                    }

        path = 'config/{}/'.format(configfile)
        if not os.path.exists(path):
            os.makedirs(path)
        name = path + str(count) + '.yaml'
        yamlPath = os.path.join(fileNamePath, name)

        f = open(yamlPath,'w')
        print(yaml.dump(aproject, f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', '-c', default='config', type=str, help='config file path')
    args = parser.parse_args()
    generate(args.configfile)