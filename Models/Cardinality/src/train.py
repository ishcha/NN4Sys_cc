import argparse
import time
import os
import pytorch_warmup as warmup
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data import get_train_datasets, load_data, make_dataset
from mscn.model import SetConv




def train(num_queries, num_epochs, batch_size, hid_units, cuda, save_path, materialize, lr):
    if materialize:
        # Load training and validation data
        num_materialized_samples = 1000
    else:
        num_materialized_samples = 0
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_samples, max_num_joins, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    table2vec, column2vec, op2vec, join2vec = dicts
    dicts.extend([column_min_max_vals, min_val, max_val, max_num_samples, max_num_joins, max_num_predicates])
    torch.save(dicts, os.path.join(save_path, 'saved_dicts.pt'))
    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, predicate_feats, join_feats, hid_units, max_num_samples, max_num_joins, max_num_predicates)
    print(1)




def main():

    queries = 10000
    epochs=10


    train(queries, epochs, 1024, 128, False, "./", True, 0.001)


if __name__ == "__main__":
    main()
