import pickle
import tensorflow as tf
import random
import numpy as np
random.seed(1)

BATCH_SIZE = 255
DATA_PATH = "./data/discriminator_test.pkl"
DISPLAY_EVERY = 100
LEARNING_RATE =.001
NUM_STEPS = 100000
class DataReader:
  """Assemble and return a batch of data and labels.
  
  inputs:
    batch_size: number of datapoints and labels per batch
    data_path: path to file holding datapoints (.pkl as of 3/24/19)
  
  returns:
    data: batch_size x [[obs_vec&action],[obs_vec&action]]
    labels: batch_size x [same_bool]
    
  Each datapoint in a batch consists of two observation vector + action vector
  concatenations, and a label boolean saying whether the two concatenated
  vectors were produced by the same agent. Both concat vectors in a datapoint
  are taken from the same move number (eg. both are move 32, but not from the
  same game). It is assumed that two players played the game (FIXME). 
  """
  
  def __init__(self, data_path, batch_size):
    
    # self.all_data is a dictionary with:
    # keys: the number of the move when the observation and action were taken
    # values: list of tuples [(observation vector, action vector, agent_id str), ...]
    self.all_data = pickle.load(open(data_path,"rb"))
    self.train_data = {}
    self.test_data = {}
    self.batch_size = batch_size

    for move_num in self.all_data:
      split_idx = int(0.9 * len(self.all_data[move_num]))
      self.train_data[move_num] = self.all_data[move_num][:split_idx]
      self.test_data[move_num] = self.all_data[move_num][split_idx:]
  
  def next_batch(self, train=True):
    if train:
      data_bank = self.train_data
    else:
      data_bank = self.test_data
      
    move_nums = [random.choice(list(data_bank)) for _ in range(self.batch_size)]
    data = []
    labels = []
    for move_num in move_nums:
      obs_vec1, act_vec1, agent1 = random.choice(data_bank[move_num])
      obs_vec2, act_vec2, agent2 = random.choice(data_bank[move_num])
      
      data.append([obs_vec1 + act_vec1, obs_vec2 + act_vec2])
      labels.append(int(agent1 == agent2))
    
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
  data_reader = DataReader(data_path=DATA_PATH, batch_size=BATCH_SIZE)
  obs_vector_len, action_vector_len = len(data_reader.all_data[0][0][0]), len(data_reader.all_data[0][0][1])

  graph = tf.Graph()
  with graph.as_default():
    # data of the form: batch_size x [[obs_vec&action],[obs_vec&action]]
    data = tf.placeholder(dtype=tf.float32, shape=[None, 2, obs_vector_len + action_vector_len])
    top_input, bottom_input = data[:,0,:], data[:,1,:]
    
    # labels of the form: batch_size x [same_bool]
    labels_ph = tf.placeholder(dtype=tf.int32, shape=[None])
    labels = tf.one_hot(indices=labels_ph, depth=2)

    # Discriminator structure:
    top_hidden1 = tf.contrib.layers.fully_connected(top_input, 512, activation_fn=tf.nn.relu)
    bottom_hidden1 = tf.contrib.layers.fully_connected(bottom_input, 512, activation_fn=tf.nn.relu)
    concat_hidden1 = tf.concat([top_hidden1, bottom_hidden1], axis=1)
    final_hidden = tf.contrib.layers.fully_connected(concat_hidden1, 512, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(final_hidden, 2, activation_fn=None)
    
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(labels,1), tf.argmax(logits,1)), tf.int32)), BATCH_SIZE)
    
  # running session
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    for step in range(NUM_STEPS):
      train_data, train_labels = data_reader.next_batch(train=True)
      loss_val, train_acc_val, _ = sess.run([loss, accuracy, optimizer], feed_dict={data:train_data, labels_ph:train_labels})
      
      if step % DISPLAY_EVERY == 0:
        test_data, test_labels = data_reader.next_batch(train=False)
        test_acc_val = sess.run([accuracy], feed_dict={data:test_data, labels_ph:test_labels})
        print('step {:4d}:    loss={:7.3f}    tr_acc={:.3f}    ts_acc={:.3f}'.format(step, loss_val, train_acc_val, test_acc_val[0]))
      
if __name__ == "__main__":
  main()
