import tensorflow as tf
import numpy as np
import quandl

quandl.ApiConfig.api_key = 'nuYEr_cSKvuJjstbzSzV'




def create_feature_sets_and_labels():
    df = quandl.get("WIKI/GOOGL")
    df = df[['Adj. Close']]


    df['next1'] = df['Adj. Close'].shift(-1)
    df['next2'] = df['Adj. Close'].shift(-2)
    df['next3'] = df['Adj. Close'].shift(-3)
    df['next4'] = df['Adj. Close'].shift(-4)
    df['label'] = df['Adj. Close'].shift(-5)

    df.dropna(inplace=True)
    test_part = int(len(df)*0.02)

    input = df[['Adj. Close','next1','next2','next3','next4']].values.tolist()
    output = df[['label']].values.tolist()

    train_x = input[:-test_part]
    train_y = output[:-test_part]
    test_x = input[-test_part:]
    test_y = output[-test_part:]

    return train_x,train_y,test_x,test_y


train_x, train_y, test_x, test_y = create_feature_sets_and_labels()


n_nodes_hl1 = 1000
n_nodes_hl2 = 1000

n_classes = 1
batch_size = 100
hm_epochs = 10

x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float',[None,n_classes])
print(len(train_y))

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}


output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }





# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        prex = sess.run(prediction, feed_dict={x: test_x})
        print(prex)


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

        pred = tf.argmax(y, 1)
        pred2 = tf.argmax(prediction, 1)
        s = pred.eval(feed_dict={x: test_x, y: test_y})
        #        print(len(test_y),len(pred.eval(feed_dict={x:test_x})))
        #        print(s)
        s2 = pred2.eval(feed_dict={x: test_x, y: test_y})
        for i in range(len(test_y)):
            print(test_y[i], s[i], s2[i])





train_neural_network(x)