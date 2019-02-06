import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    
    To allow various hyperparameter configurations,
    put code into a TextCNN class, generating the model graph in the init function.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters):
        """
        *sequence_length: The length of sentences. We padded all our sentences to have the same length (59 for our data set).
        *num_classes: Number of classes in the output layer, two in our case (positive and negative).
        *vocab_size: The size of our vocabulary. This is needed to define the size of our embedding layer,
                     which will have shape [vocabulary_size, embedding_size].
        *embedding_size: The dimensionality of our embeddings.
        *filter_sizes: The number of words we want our convolutional filters to cover.
                       We will have num_filters for each size specified here.
                       For example, [3, 4, 5] means that we will have filters that slide over 3, 4 and 5 words respectively,
                         for a total of '3 * num_filters' filters.
        *num_filters: The number of filters per filter size.
        """

        # We start by defining the input data that we pass to our network:
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        """
        tf.placeholder creates a placeholder variable that we feed to the network when we execute it at train or test time.
        The second argument is the shape of the input tensor. 'None' means that the length of that dimension could be anything.
        In our case, the first dimension is the batch size, and using None allows the network to handle arbitrarily sized batches.
        **self는 객체의 인스턴스 그 자체를 의미.
        **대부분 객체지향 언어는 이걸 메소드에 안 보이게 전달하지만 파이썬에서 클래스의 메소드를 정의할 때는
        **self를 꼭 명시해하고 그 메소드를 불러올 때 self는 자동으로 전달됨.
        """

        #1st layer: Embedding layer - maps vocabulary word indices into low-dimensional vector representations.
        #A lookup table that we learn from a data.
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        """
        *with: 파이썬에서 파일을 입/출력하려면 open() / 기록 or 읽기 / close() 3단계를 거쳐야함.
               반면 with 문을 이용하면 with 블록을 벗어나는 순간 파일이 자동으로 닫혀(close) 코드가 좀 더 간결해짐.
        *tf.device("/cpu:0"): forces an operation to be executed on the CPU.
                              By default TensorFlow will try to put the operation on the GPU if one is available,
                              but the embedding implementation doesn’t currently have GPU support and throws an error
                              if placed on the GPU.
        *tf.name_scope: creates a new Name Scope with the name “embedding”.
                        The scope adds all operations into a top-level node called “embedding”
                        so that you get a nice hierarchy when visualizing your network in TensorBoard.
            *name scope 개념 참조: https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/how_tos/variable_scope/
        *W: our embedding matrix that we learn during training. We initialize it using a random uniform distribution.
        *tf.nn.embedding_lookup: creates the actual embedding operation.
        **The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size].
        **TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions
          corresponding to batch, width, height and channel.
          The result of our embedding doesn’t contain the channel dimension, so we add it manually,
          leaving us with a layer of shape [None, sequence_length, embedding_size, 1].
        """

        # Create a convolution + maxpool layer for each filter size
        # Since we use filters of different sizes, as each conv produces tensors of different shapes,
        # we need to iterate through them, create a layer for each of them, and merge the results into one big feature vector.
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        """
        Here, W is our filter matrix and h is the result of applying the nonlinearity to the convolution output.
        Each filter slides over the whole embedding, but varies in how many words it covers.
        "VALID" padding means that we slide the filter over our sentence without padding the edges,
          performing a narrow convolution that gives us an output of shape [1, sequence_length - filter_size + 1, 1, 1].

        Performing max-pooling over the output of a specific filter size leaves us
          with a tensor of shape [batch_size, 1, 1, num_filters].
        This is essentially a feature vector, where the last dimension corresponds to our features.
        Once we have all the pooled output tensors from each filter size we combine them into one long feature vector
          of shape [batch_size, num_filters_total].
        Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible.
        """

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        """
        tf.concat(concat_dim, values, name='concat')
        텐서들의 리스트 values를 차원 concat_dim에서 이어붙임.
        """

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        """
        The fraction of neurons we keep enabled is defined by the dropout_keep_prob input to our network.
        We set this to something like 0.5 during training, and to 1 (disable dropout) during evaluation.
        """

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        """
        Using the feature vector from max-pooling (with dropout applied) we can generate predictions
          by doing a matrix multiplication and picking the class with the highest score.
        We could also apply a softmax function to convert raw scores into normalized probabilities,
          but that wouldn’t change our final predictions.
        tf.nn.xw_plus_b is a convenience wrapper to perform the 'Wx+b' matrix multiplication.
        """

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        """
        tf.nn.softmax_cross_entropy_with_logits is a convenience function that calculates the cross-entropy loss
          for each class, given our scores and the correct input labels.
        We then take the mean of the losses.
        We could also use the sum,
          but that makes it harder to compare the loss across different batch sizes and train/dev data.
        """

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
