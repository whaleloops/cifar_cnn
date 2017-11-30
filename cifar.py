import numpy as np
import os
import tensorflow as tf
import gc

from data_utils import load_tiny_imagenet

FLAGS = tf.app.flags.FLAGS


#%%
# 2.4 Build inference graph.
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, name='weight'):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name='bias'):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

def mnist_inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
    """
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(images, [-1, 64, 64, 3])

    # Convolution 1
    with tf.name_scope('conv1'):
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        W_conv1 = weight_variable([5, 5, 3, 32], 'W_conv1')
        b_conv1 = bias_variable([32], 'b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)
#         norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # Convolution 2
    with tf.name_scope('conv2'):
        # Second convolutional layer -- maps 32 feature maps to 64.
        W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv1')
        b_conv2 = bias_variable([64], 'b_conv1')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)  
#         norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # Hidden 1
    with tf.name_scope('hidden1'):
        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        tmp = h_pool2.get_shape().as_list()[1]
        W_fc1 = weight_variable([tmp * tmp * 64, hidden1_units], 'W_fc1')
        b_fc1 = bias_variable([1024], 'b_fc1')
        norm2_flat = tf.reshape(h_pool2, [-1, tmp*tmp*64])
        h_fc1 = tf.nn.relu(tf.matmul(norm2_flat, W_fc1) + b_fc1)
        variable_summaries(W_fc1)
        variable_summaries(b_fc1)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection("prop", keep_prob)  # Remember this Op.
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Hidden 2
    with tf.name_scope('hidden2'):
        # Map the 1024 features to 10 classes, one for each digit
        W_fc2 = weight_variable([hidden1_units, hidden2_units], 'W_fc2')
        b_fc2 = bias_variable([hidden2_units], 'b_fc2')
        variable_summaries(W_fc2)
        variable_summaries(b_fc2)
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

#%%
# 2.5 Build training graph.
def mnist_training(logits, labels, learning_rate):
    """Build the training graph.

    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
          range [0, NUM_CLASSES).
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
#     train_op = optimizer.minimize(loss, global_step=global_step)
    tvars = tf.trainable_variables() 
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    # Uncomment the following line to see what we have constructed.
    # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
    #                      "/tmp", "train.pbtxt", as_text=True)
    
    return train_op, loss

class DataSet(object):
  def __init__(self, image, label):
    self.index_in_epoch = 999999
    self.x = image
    self.y = label 
    self.num_examples = image.shape[0]
    self.epoch_completed = 0
  def next_batch(self, batch_size):
    start = self.index_in_epoch
    self.index_in_epoch += batch_size
    if self.index_in_epoch > self.num_examples:
        self.epoch_completed += 1
        # Shuffle the data
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.y = self.y[perm]
        # Start next epoch
        start = 0
        self.index_in_epoch = batch_size
        assert batch_size <= self.num_examples
    end = self.index_in_epoch
    return self.x[start:end], self.y[start:end]

def main(argv):
  #%% load data
  data = load_tiny_imagenet('datasets/tiny-imagenet-100-A', subtract_mean=True)
  print data['X_train'].shape

  mnist_graph = tf.Graph()
  with mnist_graph.as_default():
      # Generate placeholders for the images and labels.
      images_placeholder = tf.placeholder(tf.float32)
      labels_placeholder = tf.placeholder(tf.int32)
      tf.add_to_collection("images", images_placeholder)  # Remember this Op.
      tf.add_to_collection("labels", labels_placeholder)  # Remember this Op.


      # Build a Graph that computes predictions from the inference model.
      logits, keep_prob = mnist_inference(images_placeholder,
                               FLAGS.HIDDEN1_UNITS,
                               FLAGS.HIDDEN2_UNITS)
      tf.add_to_collection("logits", logits)  # Remember this Op.

      # Add to the Graph the Ops that calculate and apply gradients.
      train_op, loss_op = mnist_training(logits, labels_placeholder, FLAGS.lr)

      # Add the variable initializer Op.
      init = tf.global_variables_initializer()

      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

      tf.summary.scalar("Cost", loss_op)
      # Uncomment the following line to see what we have constructed.
      # tf.train.write_graph(tf.get_default_graph().as_graph_def(),
      #                      "/tmp", "complete.pbtxt", as_text=True)

  with tf.Session(graph=mnist_graph) as sess:    
      # Merge all the tf.summary
      summary_op = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter('model', sess.graph)
      
      # Run the Op to initialize the variables.
      sess.run(init)
      
      # Start the training loop.
      train_ds = DataSet(data['X_train'], data['y_train'])
      max_steps = int(FLAGS.MAX_STEPS)
      for step in xrange(max_steps):
          images, labels = train_ds.next_batch(256)
          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          _, loss, summary = sess.run([train_op, loss_op, summary_op], 
                                       feed_dict={images_placeholder: images,
                                                  labels_placeholder: labels,
                                                  keep_prob: 1.0})
          train_writer.add_summary(summary, step)
          if step % 100 == 0:  # Record execution stats
              print('Step %d/%d -- loss: %f' % (step, max_steps, loss))

      # Write a checkpoint.
      train_writer.close()
      checkpoint_file = os.path.join(FLAGS.MODEL_SAVE_PATH, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=step)
  print train_ds.epoch_completed


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('MAX_STEPS', 10000, 'abc')
    tf.app.flags.DEFINE_float('lr', 0.0001, 'abc')
    tf.app.flags.DEFINE_integer('HIDDEN1_UNITS', 1024, 'abc')
    tf.app.flags.DEFINE_integer('HIDDEN2_UNITS', 100, 'the version of model to be used')
    tf.app.flags.DEFINE_string('MODEL_SAVE_PATH', 'model', 'path_to_save_model')

    tf.app.run()