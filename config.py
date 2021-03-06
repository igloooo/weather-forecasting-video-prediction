#-*- coding: utf-8 -*-
import argparse
import pickle


def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--num_steps', type=int, default=150, help='')
net_arg.add_argument('--cell_size', type=int, default=700, help='')
net_arg.add_argument('--hyper_size', type=int, default=400, help='')
net_arg.add_argument('--embed_size', type=int, default=128, help='')
net_arg.add_argument('--hidden_size', type=int, default=256, help='')
net_arg.add_argument('--num_layers', type=int, default=2, help='')
net_arg.add_argument('--fast_layers', type=int, default=2, help='')
net_arg.add_argument('--zoneout_c', type=float, default=0.5, help='')
net_arg.add_argument('--zoneout_h', type=float, default=0.9, help='')
net_arg.add_argument('--keep_prob', type=float, default=0.65, help='')
net_arg.add_argument('--input_dim', type=int, default=300, help='')
net_arg.add_argument('--num_glimpse', type=int, default=1, help='')
net_arg.add_argument('--use_terminal_symbol', type=str2bool, default=True, help='Not implemented yet')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--task', type=str, default='ptb')
data_arg.add_argument('--batch_size', type=int, default=128)
data_arg.add_argument('--vocab_size', type=int, default=50)
data_arg.add_argument('--input_size', type=int, default=300)
data_arg.add_argument('--min_data_length', type=int, default=5)
data_arg.add_argument('--max_data_length', type=int, default=80)
data_arg.add_argument('--train_num', type=int, default=1000000)
data_arg.add_argument('--valid_num', type=int, default=1000)
data_arg.add_argument('--test_num', type=int, default=1000)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')

train_arg.add_argument('--max_epoch', type=int, default=200, help='')
train_arg.add_argument('--max_max_epoch', type=int, default=200, help='')


train_arg.add_argument('--max_step', type=int, default=1000000, help='')
train_arg.add_argument('--init_scale', type=float, default=0.002, help='')
train_arg.add_argument('--lr_start', type=float, default=0.01, help='')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='')
train_arg.add_argument('--lr_decay_rate', type=float, default= 0.1, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=1.0, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=2, help='')
misc_arg.add_argument('--num_log_samples', type=int, default=3, help='')
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'], help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--data_path', type=str, default='/Ujjawal/fast-slow-lstm/data' )
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--random_seed', type=int, default=123, help='')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config
