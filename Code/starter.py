from Training import train

data_dir = '../../Files/Delay'
epochs = [1, 10]
units = 8
lr = 3e-4
b_size = 600
model = 'ED'


dataset = 'CL1BTapePreamp'
cond = 1
train(data_dir=data_dir,
      save_folder=model+dataset+'TCP',
      dataset=dataset,
      b_size=b_size,
      learning_rate=lr,
      cond=cond,
      units=units,
      epochs=epochs,
      model=model,
      inference=False)