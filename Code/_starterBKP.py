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


from Training import train


"""
main script

"""
# data_dir: the directory in which datasets are stored
data_dir = '../'

epochs = 200
units = 8 # number of model's units
batch_size = 600 # batch size
lr = 3e-4 # initial learning rate

model = 'ED'
model_save_dir = '../weights'
#[CL1BTapePreamp, TapePreamp, CL1BTape, CL1BPreamp]

dataset = 'TapePreamp' # dataset to use

if dataset == 'CL1BTapePreamp':
      cond = 3
else:
      cond = 1
train(data_dir=data_dir,
      model_save_dir=model_save_dir,
      save_folder=model+dataset+'',
      dataset=dataset,
      batch_size=batch_size,
      learning_rate=lr,
      cond=cond,
      units=units,
      epochs=epochs,
      model=model,
      inference=True)


