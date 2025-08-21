# Copyright (C) 2024 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2024, "Hybrid Neural Audio Effects" in proceedings of Sound and Music Computing, Porto, Portugal.


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Add, LSTM, Multiply
from tensorflow.keras.layers import Lambda
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
        film = Dense(units * 2)(cond_inputs)
        g, b = Lambda(lambda x: tf.split(x, 2, axis=-1))(film)
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
