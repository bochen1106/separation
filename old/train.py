from my_reader import Reader
from my_model import MyModel
import model

import time
import logging

import operator

#from progress.bar import Bar

import tensorflow as tf
from keras.callbacks import TensorBoard
from util import Config
import os.path as osp
import numpy as np
import os


config = Config("config.json")
AUD_INPUT_DIM = config.get("AUD_INPUT_DIM")
NUM_SPK = config.get("NUM_SPK")
BATCH_SIZE_TRAIN = config.get("BATCH_SIZE_TRAIN")
BATCH_SIZE_EVAL = config.get("BATCH_SIZE_EVAL")
BATCH_SIZE_TEST = config.get("BATCH_SIZE_TEST")

#def write_log(callback, names, logs, batch_no):
#    """ Function to generate the tensorboard display.
#    """
#    for name, value in zip(names, logs):
#        summary = tf.Summary()
#        summary_value = summary.value.add()
#        summary_value.simple_value = value
#        summary_value.tag = name
#        callback.writer.add_summary(summary, batch_no)
#        callback.writer.flush()


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.avg_time = 0
        self.n_toc = 0

    def tic(self):
        self.n_toc = 0
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.n_toc += 1.
        self.avg_time = self.total_time / self.n_toc
        return self.total_time


class Logger:
    """
    When receiving a message, first print it on screen, then write it into log file.
    If save_dir is None, it writes no log and only prints on screen.
    """
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)

            self.handler = logging.FileHandler(osp.join(save_dir, 'experiment.log'))

            self.handler.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt='%(asctime)s |  %(message)s')
            self.handler.setFormatter(fmt=formatter)

            self.logger.addHandler(self.handler)

        else:
            self.logger = None

    def info(self, msg):
        print msg
        if self.logger is not None:
            self.logger.info(msg)

    def stop(self):
        self.logger.removeHandler(self.handler)
        del self.handler
        del self.logger


def eval_loss(rnn_model, data):
    """ Evaluate the model on the validation or testing sets.
    """
    loss = []
#    num_readers = len(data_models)
    """ Step 1. Make all the data_flow's the same
    """
#    for reader_index in range(1, num_readers):
#        data_models[reader_index].data_flow = data_models[0].data_flow

    """ Step 2. Set the 'index_start' for each reader
    """
    batches_per_reader = len(data.data_flow) / BATCH_SIZE_EVAL
#    for i in range(num_readers):
#        data_models[i].index_start = i * batches_per_reader * BATCH_SIZE_EVAL

    """ Step 3. Evaluate
    """
#    bar = Bar('Processing', max=batches_per_reader)
    for batch_index in range(batches_per_reader):

        x, t = data.iterate_batch()
        loss_eval = rnn_model.test_on_batch(x=[x], y=[t])
        loss.append(np.sqrt(loss_eval / ((x.shape[1] * AUD_INPUT_DIM) ** 2)))
#        bar.next()
#    bar.finish()
    # Reset for next evaluation
    data.reset()

    return np.mean(loss)


def do_rnn_train(model_path, num_layers=4, hid_dim=300,
                 pretrained_model=None, config=None):
    """ Define the dpc traininig process.
    """

    if not osp.exists(osp.join(model_path, 'code_bak')):
        os.makedirs(osp.join(model_path, 'code_bak'))
    os.system('cp *.py {}/code_bak'.format(model_path))
    os.system('cp ../*.py {}/code_bak'.format(model_path))

    """ 1. Data model
    """
    file_data = "data/h5/train/vn-cl.h5"
    file_info = "data/h5/train/vn-cl.cpickle"
    data_train = Reader(file_data=file_data, file_info=file_info, config=config)
    file_data = "data/h5/valid/vn-cl.h5"
    file_info = "data/h5/valid/vn-cl.cpickle"   
    data_valid = Reader(file_data=file_data, file_info=file_info, config=config)

    epochs = 10
    patience = 5  # Each time we get a better validation result, we increase 'patience' epochs

    iters_per_epoch = (len(data_train.data_flow) // BATCH_SIZE_TRAIN) + 1
    maxIters = epochs * iters_per_epoch
    print('Train for %d-epochs, %d-iters ...' % (epochs, maxIters))

    # Initialize the logger and timer
    logger = Logger(save_dir=model_path)
    timer = Timer()

#    tf_board_path = osp.join(model_path, 'graph') 05/03/2018
#    train_names = ['train_loss'] 05/03/2018
#    valid_names = ['valid_loss'] 05/03/2018

    """ 2. Audio model
    """
    logger.info('Building the model ...')

    audio_model = model.build_model(path_weights=pretrained_model, config=config)

    """ 3. Set params for training
    """
    logger.info('Now the params ...')
    validFreq = iters_per_epoch // 5  # validation about once an epoch
    # validFreq = 2
    dispFreq = iters_per_epoch // 25  # display 5 times each epoch
    # dispFreq = 1
    best_val_loss = float('inf')

    # Optimization
    history_loss_train = []
    history_loss_train_avg = []
    history_loss_val = []

    start_time = time.time()
    iters = 1

    best_iter = 0  # The iteration that the best model occurs
#    callback = TensorBoard(tf_board_path) 05/03/2018
#    callback.set_model(audio_model) 05/03/2018
    try:
        timer.tic()
        while iters < maxIters:
            # time0 = time.time()

            x_train, t_train = data_train.iterate_batch()
            # time1 = time.time()
            print "read a batch"
            
            loss = audio_model.train_on_batch(x=[x_train], y=[t_train])
            loss = np.sqrt(loss / ((x_train.shape[1] * AUD_INPUT_DIM) ** 2))

            history_loss_train.append(loss)
            history_loss_train_avg.append(np.mean(history_loss_train))

            # time2 = time.time()
            # print('flag%d, data: %.2f, model: %.2f ...' % (train_flag, time1 - time0, time2 - time1))

            """ Write to tensorboard
            """
            # 05/03/2018
#            tf_board_train = []
#            tf_board_train.append(history_loss_train_avg[-1])
#            write_log(callback, train_names, tf_board_train, iters)

            if np.mod(iters, dispFreq) == 0:
                logger.info('iter={}, training loss = {:.5f}, finish -> {}, time={:.1f} sec'
                            .format(iters, history_loss_train_avg[-1], maxIters, timer.toc()))
                timer.tic()

            if np.mod(iters, validFreq) == 0:
                print '... Computing validation err'
                val_loss = eval_loss(rnn_model=audio_model, data=data_valid)

                history_loss_val.append(val_loss)

                logger.info(msg='--- this valid loss = {:.4f}, best = {:.4f}'.format(val_loss, best_val_loss))
                model_file = model_path + '/model.iter' + str(iters)
                logger.info(msg='Saving model at iter={}'.format(iters))
#                model.save_model(model=audio_model, filename=model_file)
                

                # Write to tensorboard
                # 05/03/2018
#                tf_board_valid = []
#                tf_board_valid.append(val_loss)
#                write_log(callback, valid_names, tf_board_valid, iters // validFreq)

                # save the model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    maxIters = max(maxIters, iters + patience * iters_per_epoch)

                    model_file = model_path + '/model.best'
                    logger.info(msg='Saving model at iter={}'.format(iters))
                    logger.info(msg='Saving best model at iter={}'.format(iters))
                    model.save_model(model=audio_model, filename=model_file)
#                    m.save_model(filename=model_file)

            iters += 1
    except KeyboardInterrupt:
        logger.info('Training interrupted ...')
    end_time = time.time()

    # Close the valid iterators
    data_valid.close()

    # save the train costs
    train_cost_file = model_path + '/loss.npz'
    np.savez(train_cost_file, history_loss_train=history_loss_train,
             history_loss_train_avg=history_loss_train_avg,
             history_loss_val=history_loss_val,
             time=(end_time - start_time) / 60.,
             data_mean=data_train.data_mean,  # mean of the training data
             data_std=data_train.data_std,
             best_iter=best_iter,
             validFreq=validFreq)  # std of the training data

    data_train.close()

    # Test
    data_test = Reader(file_data="data/h5/test/vn-cl.h5", 
                                                 file_info="data/h5/test/vn-cl.cpickle", 
                                                 config=config)
           
    audio_model.load_weights(model_path + '/model.best.h5')
    test_loss = eval_loss(rnn_model=audio_model, data=data_test)
    data_test.close()
    logger.info(msg='--- Test loss = {:.4f}'.format(test_loss))
    logger.info(msg='Training done ...')

    # Close the reader
    logger.info(msg='Shutting down the reader ...')
    logger.info(msg='Training took {:.2f} minutes in total ...'.format((end_time - start_time) / 60.))
    logger.stop()

#%%
do_rnn_train("data/model2/", num_layers=4, hid_dim=300,
                 pretrained_model=None, config=config)