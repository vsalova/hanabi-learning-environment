import pickle
import random
random.seed(1)

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
  
  def __init__(data_path, batch_size)
    
    # self.all_data is a dictionary with:
    # keys: the number of the move when the observation and action were taken
    # values: list of tuples [(observation vector, action vector, agent enum), ...]
    self.all_data = pickle.load(open(data_path,"rb"))
  
  def next_batch():
    move_nums = [random.choice(list(self.all_data)) for _ in range(batch_size)]
    data = []
    labels = []
    
    for move_num in move_nums:
      obs_vec1, act_vec1, agent1 = random.choice(self.all_data[move_num])
      obs_vec2, act_vec2, agent2 = random.choice(self.all_data[move_num])
      
      data.append([obs_vect1 + act_vec1], [obs_vec2 + act_vec2])
      labels.append(agent1 == agent2)
    
    return data, labels


def main():
  with graph.as_default():
    # assemble the graph here
    
    # data of the form: batch_size x [[obs_vec&action],[obs_vec&action]]
    data = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, 2, \
                          len(obs_vector) + len(action_vector)))
    
    # labels of the form: batch_size x [same_bool]
    labels = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE))
    lables = tf.one_hot() #FIXME: implement
    
    top_input, bottom_input = data[:,0,:], data[:,1,:]
    
    # Discriminator structure:
    top_hidden1 = tf.contrib.layers.fully_connected(top_input, 512, activation_fn=tf.nn.relu)
    bottom_hidden1 = tf.contrib.layers.fully_connected(bottom_input, 512, activation_fn=tf.nn.relu)
    concat_hidden1 = tf.concat([top_hidden1, bottom_hidden1], axis=0)
    final_hidden = tf.contrib.layers.fully_connected(concat_hidden1, 512, activation_fn=tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(final_hidden, 2, activation_fn=None)
    
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    accuracy = # calculate accuracy of prediction
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    
  # running session
  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    data_reader = DataReader(data_path=DATA_PATH, batch_size=BATCH_SIZE)
    
    for step in range(NUM_STEPS):
      train_data, train_labels = data_reader.next_batch()
      loss_val, acc_val, _ = sess.run([loss, accuracy, optimizer], feed_dict={data:train_data, labels:train_labels})
      print('step {:4d}:    loss={:7.3f}    acc={:.3f}'.format(step, loss_val, acc_val))
      
if __name__ == "__main__":
  main()