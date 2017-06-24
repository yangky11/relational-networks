import argparse
import os.path
import torch


def parseArgs():

  parser = argparse.ArgumentParser(description='Relation Networks')

  parser.add_argument('--model', type=str, default='RN', choices=['RN', 'CNN-MLP'],
                      help='the model to use')
  parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--epochs', type=int, default=20, metavar='N',
                      help='number of epochs to train (default: 20)')
  parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                      help='learning rate (default: 0.0001)')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--log_dir', type=str, default='./logs/')
  parser.add_argument('--exp_id', type=str, default='experiment')
  parser.add_argument('--checkpoint', type=str,
                      help='resume from model stored')

  args = parser.parse_args()
 
  args.log_dir = os.path.join(args.log_dir, args.exp_id)
  args.checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
  if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
  if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
    

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed) 

  return args
