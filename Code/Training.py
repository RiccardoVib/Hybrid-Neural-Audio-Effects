import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_ED, create_model
from DatasetsClass import DataGeneratorPickles
import numpy as np
import random
from Metrics import ESR, STFT_t, STFT_f, RMSE, flux
import sys
import time


def train(**kwargs):
    b_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    model_name = kwargs.get('model', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', [1, 60])
    D = kwargs.get('cond', 0)

    epochs0 = epochs[0]
    epochs1 = epochs[1]
    start = time.time()

    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    fs = 44100

    e = 32
    d = 32
    w = e + d

    model = create_model_ED(cond_dim=D, input_dim=w, units=units, b_size=b_size)
    #model = create_model(cond_dim=D, input_dim=w, units=units, b_size=b_size)

    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)

    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]

        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

        # load the datasets
        train_gen = DataGeneratorPickles(data_dir, dataset + '_train.pickle', input_size=w,
                                         cond_size=D, batch_size=b_size)
        val_gen = DataGeneratorPickles(data_dir, dataset + '_val.pickle', input_size=w,
                                       cond_size=D, batch_size=b_size)
        training_steps = train_gen.training_steps
        opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps, epochs0, 4),
                                       clipnorm=1)
        losses = 'mse'
        model.compile(loss=losses, optimizer=opt)

        loss_training = []
        loss_val = []

        # train the model
        for i in range(epochs0, epochs1, 1):
            print('epochs:', i)
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=val_gen,
                                callbacks=callbacks)
            loss_training.append(results.history['loss'])
            loss_val.append(results.history['val_loss'])

        # save results
        writeResults(results, units, epochs, b_size, learning_rate, model_save_dir,
                     save_folder, epochs[0])

        loss_training.append(results.history['loss'])
        loss_val.append(results.history['val_loss'])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder)

        print("Training done")

    avg_time_epoch = (time.time() - start)

    sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")

    sys.stdout.write("\n")
    sys.stdout.flush()

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    # compute test loss
    test_gen = DataGeneratorPickles(data_dir, dataset + '_val.pickle', input_size=w,
                                    cond_size=D, batch_size=b_size)
    model.reset_states()
    predictions = model.predict(test_gen, verbose=0)[:, 0]
    predictWaves(predictions, test_gen.x[w:len(predictions) + w], test_gen.y[w:len(predictions) + w], model_save_dir,
                 save_folder, fs, '0')

    mse = tf.keras.metrics.mean_squared_error(test_gen.y[w:len(predictions) + w], predictions)
    mae = tf.keras.metrics.mean_absolute_error(test_gen.y[w:len(predictions) + w], predictions)
    esr = ESR(test_gen.y[w:len(predictions) + w], predictions)
    rmse = RMSE(test_gen.y[w:len(predictions) + w], predictions)
    sftf_t = STFT_t(test_gen.y[w:len(predictions) + w], predictions)
    sftf_f = STFT_f(test_gen.y[w:len(predictions) + w], predictions)
    spectral_flux = flux(test_gen.y[w:len(predictions) + w], predictions, fs)

    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'spectral_flux': spectral_flux, 'sftf_t': sftf_t,
                'sftf_f': sftf_f}

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + str(dataset) + 'results.txt'])),
              'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)

    return 42