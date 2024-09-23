from Training import train


"""
main script

"""
# data_dir: the directory in which datasets are stored
data_dir = '../../Files/'

epochs = 200
units = 8 # number of model's units
batch_size = 600 # batch size
lr = 3e-4 # initial learning rate


model = 'ED'

dataset = 'CL1BTapePreamp' # dataset to use
cond = 1
train(data_dir=data_dir,
      save_folder=model+dataset+'TCP',
      dataset=dataset,
      batch_size=batch_size,
      learning_rate=lr,
      cond=cond,
      units=units,
      epochs=epochs,
      model=model,
      inference=False)
