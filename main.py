"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
import argparse
import os
import pickle
import random
import numpy as np
import torch
from torch.autograd import Variable
from parseArgs import parseArgs
from model import RN, CNN_MLP


args = parseArgs()

if args.model == 'RN':
  model = RN(args)
elif args.model == 'CNN-MLP':
  model = CNN_MLP(args)
model.cuda()

batchsize = args.batchsize
input_img = Variable(torch.FloatTensor(batchsize, 3, 75, 75).cuda())
input_qst = Variable(torch.FloatTensor(batchsize, 10).cuda())
label = Variable(torch.LongTensor(batchsize).cuda())


def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][batchsize * i : batchsize * (i + 1)]))
    qst = torch.from_numpy(np.asarray(data[1][batchsize * i : batchsize * (i + 1)]))
    ans = torch.from_numpy(np.asarray(data[2][batchsize * i : batchsize * (i + 1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    for batch_idx in range(len(rel[0]) // batchsize):
        tensor_data(rel, batch_idx)
        accuracy_rel = model.train_(input_img, input_qst, label)

        tensor_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(epoch, batch_idx * batchsize * 2, len(rel[0]) * 2, \
                                                                                                                           100. * batch_idx * batchsize/ len(rel[0]), accuracy_rel, accuracy_norel))
            

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // batchsize):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))

    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    train_datasets, test_datasets = pickle.load(open(filename, 'rb'))
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)
    

rel_train, rel_test, norel_train, norel_test = load_data()

if args.checkpoint:
  print('==> loading checkpoint {}'.format(args.checkpoint))
  model.load_state_dict(torch.load(args.checkpoint))

for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train, norel_train)
    test(epoch, rel_test, norel_test)
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model_%d.pth' % epoch))
