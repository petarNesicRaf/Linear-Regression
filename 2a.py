# TODO popuniti kodom za problem 2a
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#polinomijalna matrica
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

#plt.scatter(data['x'], data['y'])
#plt.xlabel('funky x')
#plt.ylabel('funky y')

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
#stepen polinoma 6
nb_features = 6

data['x'] = create_feature_matrix(data['x'], nb_features)

w = tf.Variable(tf.zeros(nb_features))
b = tf.Variable(0.0)

learning_rate = 0.001
#zbog testiranja 50
nb_epochs = 50


def pred(x, w, b):
  w_col = tf.reshape(w, (nb_features, 1))
  hyp = tf.add(tf.matmul(x, w_col), b)
  return hyp

#mse
def loss(x, y, w, b, reg = None):
  prediction = pred(x, w, b)

  y_col = tf.reshape(y, (-1, 1))
  mse = tf.reduce_mean(tf.square(prediction - y_col))
  
  lmbd = 0.01

  if reg == 'l1':
      l1_reg = lmbd * tf.reduce_mean(tf.abs(w))
      loss = tf.add(mse, l1_reg)
  elif reg == 'l2':
      l2_reg = lmbd * tf.reduce_mean(tf.square(w))
      loss = tf.add(mse, l2_reg)
  else:
      loss = mse

  return loss

#grad
def gradient(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')
    
    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def train(x, y, w, b):
    w_grad, b_grad, loss = gradient(x, y, w, b)

    adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

loss_function = []

for epoch in range(nb_epochs):
    
    epoch_loss = 0
    for sample in range(nb_samples):
        x = data['x'][sample].reshape((1, nb_features))
        y = data['y'][sample]

        curr_loss = train(x, y, w, b)
        epoch_loss += curr_loss

    epoch_loss /= nb_samples
    loss_function.append(epoch_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')
    

print(f'w = {w.numpy()}, bias = {b.numpy()}')
#xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'), nb_features)

xs = np.linspace(min_x, max_x, 100, dtype='float32')
xs_feature_matrix = create_feature_matrix(xs, nb_features)
hyp_val = pred(xs_feature_matrix, w, b)

ys = []


for i in range(nb_features):
    w_col = tf.reshape(tf.gather(w, i), (1, 1))
    xs_feature_col = tf.reshape(xs_feature_matrix[:, i], (-1, 1))
    ys.append(tf.matmul(xs_feature_col, w_col))

ys = tf.concat(ys, axis=1)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

#plotovanje krivih
axs[0].set_xlabel('funky x')
#plotovanje hipoteze
axs[0].plot(xs_feature_matrix[:,0].tolist(), hyp_val.numpy().tolist(), color='r')
axs[0].set_ylabel('funky y')
axs[0].plot(xs, ys)
axs[0].scatter(dat['x'], data['y'], label='data')

#plotovanje gubitka
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].plot(np.arange(nb_epochs), loss_function)


axs[0].legend()
axs[0].set_xlim([-3,3])
axs[0].set_ylim([-3,3])


plt.tight_layout()
plt.savefig('2A_grafik')
plt.show()
