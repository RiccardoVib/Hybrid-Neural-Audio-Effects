import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Add, LSTM, Multiply

from Layers import GLU


def create_model_ED(D, T, units, batch_size=600):
    """ 
    ED model
    :param T: input size
    :param D: number of conditioning parameters
    :param units: number of units
    :param batch_size: batch size
    """

    encoder_inputs = Input(batch_shape=(batch_size, T - 1, 1), name='encoder_input')
    encoder_outputs, h, c = LSTM(units, stateful=True, return_sequences=False, return_state=True, name='LSTM_encoder')(encoder_inputs)

    decoder_inputs = Input(shape=(1, 1), name='decoder_input')

    decoder_outputs = LSTM(units, return_sequences=False, return_state=False, name='LSTM_decoder')(
        decoder_inputs, initial_state=[h, c])

    if D != 0:
        cond_inputs = Input(shape=(D,), name='conditioning_input')
        film = Dense(units * 2, batch_input_shape=(batch_size, units))(cond_inputs)
        g, b = tf.split(film, 2, axis=-1)
        decoder_outputs = Multiply()([decoder_outputs, g])
        decoder_outputs = Add()([decoder_outputs, b])
        decoder_outputs = GLU(in_size=units)(decoder_outputs)

        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        model = tf.keras.Model([cond_inputs, encoder_inputs, decoder_inputs], decoder_outputs)
    else:
        decoder_outputs = Dense(1, name='OutLayer')(decoder_outputs)
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.summary()

    return model
