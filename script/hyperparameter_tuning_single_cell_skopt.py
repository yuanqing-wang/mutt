"""
hyperparameter_tunning.py

MIT License

Copyright (c) 2018

Weill Cornell Medicine, Memorial Sloan Kettering Cancer Center,
Nicea Research, and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# =============================================================================
# imports
# =============================================================================
# infrastructure
from __future__ import absolute_import
from skopt import gp_minimize
from skopt.space import Categorical
import tensorflow as tf
# import horovod.tensorflow as hvd
# hvd.init()
# config = tf.ConfigProto()
# config.gpu_options.visible_device_list = str(hvd.local_rank())
tf.enable_eager_execution()
from sklearn.metrics import r2_score
from Bio import SeqIO
import pyBigWig
import numpy as np
import pandas as pd
import uuid
import os
import re
import tempfile
import shutil
import multiprocessing

# module
from mutt import *

# =============================================================================
# constants
# =============================================================================
N_CPUS = multiprocessing.cpu_count()
TRANSLATION = {
    'A': 0,
    'T': 1,
    'C': 2,
    'G': 3,
    'a': 0,
    't': 1,
    'c': 2,
    'g': 3,
    'N': 4,
    'n': 4,
    'O': 4}


# =============================================================================
# utility functions
# =============================================================================
def read_chr(fasta_dir, bigwig_dir, window_width):
    """ read the sequence and accesibility of one chromosome

    Parameters
    ----------
    record : SeqIO.record object
    bw : pyBigWig object

    Returns
    -------
    ds : tf.data.Dataset
        the resulting dataset
    """
    chr_name = fasta_dir.split('.')[0]

    # get the record
    record = SeqIO.read(fasta_dir, 'fasta')

    # read the accesibility data
    bw = pyBigWig.open(bigwig_dir)

    # handle the idxs
    seq_len = len(record)
    assert seq_len == int(re.findall(str(bw.chroms(chr_name))[0]))
    idxs = range(len(seq_len - window_width))
    idxs_ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(idxs))

    # define map function
    def map_func(idx):
        # handle x
        x = str(record.seq[idx, idx + window_width])
        x = np.array([TRANSLATION[ch] for ch in list(x)],
            dtype=np.int32)
        x = tf.one_hot(
            tf.convert_to_tensor(x),
            5,
            dtype=tf.float32)

        # handle y
        y = bw.values(chr_name, idx, idx+window_width)
        y = tf.convert_to_tensor(y)
        return x, y

    ds = idxs_ds.map(map_func,
        num_parallel_calls=N_CPUS)

    return ds

def read_chrs(bigwig_dirs, fasta_dirs, window_width):
    """ read data on multiple fasta and sequences
    """
    ds = tf.data.Dataset()

    # loop through all the bigwig dirs and all the fasta dirs
    for bigwig_dir in bigwig_dirs:
        for fasta_dir in fasta_dirs:
            ds = ds.concatenate(read_chr(bigwig_dir, fasta_dir))

    return ds

# =============================================================================
# define trainable
# =============================================================================
class Flow(object):
    """
    training and evaluating a model with certain configuration

    """

    def build(self, single_config_values):
        """
        build the models with parameters

        Parameters
        ----------
        config : dict
            configuration of hyperparameters
        """
        global config_keys

        config = dict(zip(config_keys, single_config_values))
        print(config)
        self.encoder1 = conv.ConvNet([
                'C_%s_%s' % (
                    config['conv1_unit'],
                    config['conv1_kernel_size']),

                config['conv1_activation'],

                'C_%s_%s' % (
                    config['conv2_unit'],
                    config['conv2_kernel_size']),

                config['conv2_activation'],

                'D_%s' % config['dropout2'],

                'C_%s_%s' % (
                    config['conv3_unit'],
                    config['conv3_kernel_size']),

                config['conv3_activation'],

                'D_%s' % config['dropout3'],
                ])

        self.attention = attention.Attention(
            config['attention_units'],
            config['attention_head'])



        optimizer = tf.train.AdamOptimizer(
            float(config["learning_rate"]))

        self.optimizer = optimizer
        # self.optimizer = hvd.DistributedOptimizer(optimizer)

    def _setup(self, config_values):
        """
        define models with parameters

        Parameters
        ----------
        config : dict
            configuration of hyperparameters
        """

        global df

        # init iteration
        self.iteration = 0
        self.build(config_values)

        # get sample size
        self.n_sample = df.shape[0]

        # get ds
        x_all = np.array(
            [[TRANSLATION[ch] for ch in list(seq)] for seq \
                in df.values[:, -1]\
                .flatten().tolist()],
            dtype=np.int32)
        y_all = np.array(df.values[:, 8:-1],
            dtype=np.float32)

        # one-hot encoding
        x_all = tf.one_hot(tf.convert_to_tensor(x_all), 5,
            dtype=tf.float32)
        y_all = tf.convert_to_tensor(y_all, dtype=tf.float32)


        # normalize
        y_mean, y_var = tf.nn.moments(y_all, axes=[0])
        y_all = tf.div(y_all - y_mean, tf.sqrt(y_var))

        # put into ds
        ds = tf.data.Dataset.from_tensor_slices((x_all, y_all))
        # ds = ds.batch(128, True)
        # ds = ds.shuffle(y_tr.shape[0])
        self.ds = ds # point ds to class ref


    def _train(self):
        """
        training, and test using 5-fold cross validation

        """
        global df

        # init r2
        r2s = []

        # =====================================================================
        # data preparation
        # =====================================================================

        # split
        n_te = int(0.2 * self.n_sample)
        ds = self.ds.shuffle(self.n_sample)

        # five fold cross-validation
        for idx in range(5):
            # test: [idx * n_te: (idx + 1) * n_te]
            # train : [0: idx * n_te, (idx + 1) * n_te:]
            ds_tr = ds.take(idx * n_te).concatenate(
                ds.skip((idx + 1) * n_te).take((4 - idx) * n_te))
            ds_te = ds.skip(idx * n_te).take((idx + 1) * n_te)

            # batch ds
            ds_tr = ds_tr.batch(128, True)
            ds_te = ds_te.batch(128, True)

            # ~~~~~~~~~~~~~~~~~
            # train for a batch
            # ~~~~~~~~~~~~~~~~~
            # enumerate

            for batch, (xs, ys) in enumerate(ds_tr):
                with tf.GradientTape(persistent=True) as tape: # grad tape
                    # flow
                    x = self.encoder1(xs)
                    x = self.encoder2(x)
                    x = self.encoder3(x)
                    x = self.attention(x, x)
                    x = self.encoder4(x)
                    y_bar = self.regression(x)
                    loss = tf.losses.mean_squared_error(ys, y_bar)
                    print(ys)
                    print(y_bar)
                    print(loss)

                # backprop
                variables = self.encoder1.variables\
                    + self.attention.variables\
                    + self.encoder2.variables\
                    + self.encoder3.variables\
                    + self.encoder4.variables\
                    + self.regression.variables

                gradients = tape.gradient(loss, variables)

                self.optimizer.apply_gradients(
                    zip(gradients, variables),
                    tf.train.get_or_create_global_step())

            # increment
            self.iteration += 1

            # ~~~~
            # test
            # ~~~~
            # init
            y_true = None
            y_pred = None
            for batch, (xs, ys) in enumerate(ds_te): # loop through test data
                x = self.encoder1(xs)
                x = self.encoder2(x)
                x = self.encoder3(x)
                x = self.attention(x, x)
                x = self.encoder4(x)
                y_bar = self.regression(x)

                # put results in the array
                if type(y_true) == type(None):
                    y_true = ys.numpy()
                else:
                    y_true = np.concatenate([y_true, ys], axis=0)

                if type(y_pred) == type(None):
                    y_pred = y_bar.numpy()
                else:
                    y_pred = np.concatenate([y_pred, y_bar], axis=0)

            for idx in range(y_true.shape[1]):
                r2s.append(r2_score(y_true[:, idx], y_pred[:, idx]))

        return np.mean(r2s)

    def _save(self, checkpoint_dir):

        """
        save the model

        Parameters
        ----------
        checkpoint_dir : str
            directory to which the checkpoint is stored
        """
        # get a temp path
        dirpath = tempfile.mkdtemp()

        # save the weights
        self.encoder1.save_weights(os.path.join(dirpath,
            'models/encoder1.h5'))
        self.attention.save_weights(os.path.join(dirpath,
            'models/attention.h5'))
        self.encoder2.save_weights(os.path.join(dirpath,
            'models/encoder2.h5'))
        self.encoder3.save_weights(os.path.join(dirpath,
            'models/encoder3.h5'))
        self.encoder4.save_weights(os.path.join(dirpath,
            'models/encoder4.h5'))
        self.regression.save_weights(os.path.join(dirpath,
            'models/regression.h5'))

        # compress
        os.chdir(dirpath)
        os.system('zip -r models.zip models')
        os.system('mv models.zip ' + checkpoint_dir)

        # remove the path
        shutil.rmtree(dirpath)

        # return path
        return os.path.join(checkpoint_dir, 'models.zip')

    def _restore(self, checkpoint_path):
        """
        restore

        Parameters
        ----------
        checkpoint_path : str
            the path of checkpoint
        """
        # get a temp path
        dirpath = tempfile.mkdtemp()

        # unzip
        os.system('unzip ' + checkpoint_path + ' -d ' + dirpath)

        # init the network
        self.build()

        # save the weights
        self.encoder1.load_weights(os.path.join(dirpath,
            'models/encoder1.h5'))
        self.attention.load_weights(os.path.join(dirpath,
            'models/attention.h5'))
        self.encoder2.load_weights(os.path.join(dirpath,
            'models/encoder2.h5'))
        self.encoder3.load_weights(os.path.join(dirpath,
            'models/encoder3.h5'))
        self.encoder3.load_weights(os.path.join(dirpath,
            'models/encoder4.h5'))
        self.regression.load_weights(os.path.join(dirpath,
            'models/regression.h5'))


if __name__ == "__main__":

    # =========================================================================
    # data preparation
    # =========================================================================
    fasta_dir = 'chr1.fa' # specify fasta



    # =========================================================================
    # define search method
    # =========================================================================
    # s = lambda x: x # for population
    # s = tune.grid_search # for grid search
    s = lambda x: Categorical(x)
    # =========================================================================
    # define search space
    # =========================================================================

    config_ = {
        # ~~~~~~~~~~~~~~~~~~
        # architecture specs
        # ~~~~~~~~~~~~~~~~~~

        # conv before attention
        # conv1
        "conv1_unit": s([64, 128, 256]),
        "conv1_kernel_size": s([4, 6, 8, 10, 12]),
        "conv1_activation": s(['sigmoid', 'tanh', 'elu']),

        # attention
        "attention_units": s([64, 128, 256, 512]),
        "attention_head": s([4, 8, 16]),

        # conv after attention
        # conv2
        "conv2_unit": s([64, 128, 256]),
        "conv2_kernel_size": s([4, 6, 8, 10, 12]),
        "conv2_activation": s(['sigmoid', 'tanh', 'elu']),

        # conv3
        "conv3_unit": s([64, 128, 256]),
        "conv3_kernel_size": s([4, 6, 8, 10, 12]),
        "conv3_activation": s(['sigmoid', 'tanh', 'elu']),

        # conv4
        # with dilation
        "conv4_unit": s([64, 128, 256]),
        "conv4_kernel_size": s([4, 6, 8, 10, 12]),
        "conv4_dilation_rate": s([1, 2, 4, 8]),
        "conv4_activation": s(['sigmoid', 'tanh', 'elu']),

        # flatten layer here

        # d1
        "d1_units": s([64, 128, 256, 512, 1024]),
        "d1_activation": s(['sigmoid', 'tanh', 'elu']),
        "d1_dropout": s([0.10, 0.20, 0.30, 0.40]),

        # d1
        "d2_units": s([64, 128, 256, 512, 1024]),
        "d2_activation": s(['sigmoid', 'tanh', 'elu']),
        "d2_dropout": s([0.10, 0.20, 0.30, 0.40]),

        # ~~~~~~~~~~~~~~
        # learning specs
        # ~~~~~~~~~~~~~~
        "learning_rate": s([1e-5, 1e-4])
    }

    # seperate the dict
    config_values = config_.values()
    config_keys = config_.keys()

    def obj_func(single_config_values):
        flow = Flow()
        flow._setup(single_config_values)
        r2 = flow._train()
        return -r2

    gp_minimize(
        func=obj_func,
        dimensions=config_values,
        n_calls=1000,
        n_random_starts=10,
    )