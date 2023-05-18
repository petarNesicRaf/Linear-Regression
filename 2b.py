import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

filename = 'funky.csv'
all_data = np.loadtxt(filename, delimiter=',', dtype='float32')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]
print(data['x'][:5])
print(data['y'][:5])


#shuffle
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

#normalizacija
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

min_x, max_x = (min(data['x']), max(data['x']))
min_y, max_y = (min(data['y']), max(data['y']))

dat = dict()
dat['x'] = data['x']


nb_features = 3
data['x'] = create_feature_matrix(data['x'], nb_features)


w = tf.Variable(tf.zeros(nb_features))
b = tf.Variable(0.0)

learning_rate = 0.001
nb_epochs = 50

def pred(x, w, b):
  w_col = tf.reshape(w, (nb_features, 1))
  hyp = tf.add(tf.matmul(x, w_col), b)
  return hyp

def loss(x, y, w, b, lmbd, reg = None):
  prediction = pred(x, w, b)

  y_col = tf.reshape(y, (-1, 1))
  mse = tf.reduce_mean(tf.square(prediction - y_col))
  


  if reg == 'l1':
      l1_reg = lmbd * tf.reduce_mean(tf.abs(w))
      loss = tf.add(mse, l1_reg)
  elif reg == 'l2':
      l2_reg = lmbd * tf.reduce_mean(tf.square(w))
      loss = tf.add(mse, l2_reg)
  else:
      loss = mse

  return loss


adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train_step(x, y, w, b, lmbd):
  with tf.GradientTape() as tape:
    loss_val = loss(x, y, w, b, lmbd,reg='l2')
  
  w_grad, b_grad = tape.gradient(loss_val, [w, b])
  adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

  return loss_val

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axs[0].scatter(dat['x'], data['y'])

lambda_values = [0,0.001, 0.01, 0.1, 1, 10, 100]
loss_function = []
colors = ['red', 'yellow', 'green', 'blue', 'purple', 'black', 'brown']

for i in range(len(lambda_values)):
  color = colors[i]
  
  for epoch in range(nb_epochs):
    epoch_loss = 0
    for sample in range(nb_samples):
      x = data['x'][sample].reshape((1, nb_features))
      y = data['y'][sample]
      curr_loss = train_step(x, y, w, b, lambda_values[i])
      epoch_loss += curr_loss
    epoch_loss /= nb_samples
    loss_function.append(epoch_loss)
    if(epoch + 1) % 10 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

  print(f'Lambda = {lambda_values[i]:.3f} | w = {w.numpy()}, bias = {b.numpy()}')

  
  xs = create_feature_matrix(np.linspace(min_x, max_x,100, dtype="float32"), nb_features)
  hyp_val = pred(xs, w, b)
  
  #xs = np.linspace(min_x,max_x,100)
  #ys = np.zeros(xs.shape)
  #for j in range(nb_features):
  #  ys += w[j] * np.power(xs, j)
  
  #ax[0].plot(xs, ys, label=f"lambda={lambda_values[i]}")
  axs[0].plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), color=color)


axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
up_epochs = nb_epochs * 7
axs[1].plot(np.arange(up_epochs) , loss_function)


#ax[0].legend()
plt.show