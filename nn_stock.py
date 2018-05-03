import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
import quandl
import matplotlib.pyplot as plt
from pandas import Series

quandl.ApiConfig.api_key = 'nuYEr_cSKvuJjstbzSzV'




df = quandl.get("WIKI/TSLA")
df = df[['Adj. Close']]


df['next1'] = df['Adj. Close'].shift(-1)
df['next2'] = df['Adj. Close'].shift(-2)
df['next3'] = df['Adj. Close'].shift(-3)
df['next4'] = df['Adj. Close'].shift(-4)
df['label'] = df['Adj. Close'].shift(-5)

new_df = df[['Adj. Close','next1','next2','next3','next4','label']]
new_df.to_csv('processed.csv')

df.dropna(inplace=True)
test_part = int(len(df)*0.02)

input = np.array(df[['Adj. Close','next1','next2','next3','next4']].values.tolist())
output = np.array(df[['label']].values.tolist())

train_x = input[:-test_part]
train_y = output[:-test_part]
test_x = input[-test_part:]
test_y = output[-test_part:]

#    for x in train_y:
#        print(x)
  
n_nodes_hl1 = 10
n_nodes_hl2 = 10
#n_nodes_hl3 = 1500

n_classes = 1
batch_size = 100
hm_epochs = 40


X = tf.placeholder(tf.float32, [None, len(train_x[0])])
W = tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1], stddev=0.01))
b = tf.Variable(tf.zeros([n_nodes_hl1]))
h1 = tf.nn.relu(tf.matmul(X, W) + b)

W2 = tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_nodes_hl2]))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.random_normal([n_nodes_hl2, n_classes], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_classes]))
y = (tf.matmul(h2, W3) + b3)

Y = tf.placeholder(tf.float32, [None,n_classes])

cost = tf.reduce_mean(tf.squared_difference(y,Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#    cap = 50
    i = 0
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])

            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,
                                                          Y: batch_y})

            epoch_loss += c
            i += batch_size

        print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    #sx = np.random.rand(10,1)
#    sx = np.random.randint(cap,size=(10,1))
#    sy = np.sqrt(sx)

    print("Input")
    print(test_x)
    print("Expected Output")
    print(test_y)
    print("Predicted Output")
    predicted_output = sess.run(y,feed_dict={X: input,Y: output})
    print(sess.run(y, feed_dict={X: test_x, Y: test_y}))
    print("Error")
    print(sess.run(cost, feed_dict={X: test_x, Y: test_y}))

    predicted_output = predicted_output.flatten()
    df['Predicted'] = Series(predicted_output,index = df.index)
    df['label'].plot()
    df['Predicted'].plot()
    plt.show()

