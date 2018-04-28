import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None): 
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    TODO: Question 4 - [Application] Regression

    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.02 # adjust as necessary

        ## Layer 1
        d = 100 # end value of this layer. later can change to 300 when add more layers
        self.m = nn.Variable(1, d)
        self.b = nn.Variable(d)

        ## Layer 2
        d2 = 1 # end value of last layer must always be 1
        self.m2 = nn.Variable(d, d2)
        self.b2 = nn.Variable(d2)

    def run(self, x, y=None):
        """
        TODO: Question 4 - [Application] Regression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # graph = nn.Graph([self.m, self.b])

        ### Regression Model
        self.graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_x = nn.Input(self.graph, x)


        ## Layer 1
        xm = nn.MatrixMultiply(self.graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(self.graph, xm, self.b)
        xm_plus_b_relu = nn.ReLU(self.graph, xm_plus_b)

        ## Layer 2
        xm2 = nn.MatrixMultiply(self.graph, xm_plus_b_relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(self.graph, xm2, self.b2)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2)        

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.


            #create variables that correspond to sin graph
            #use similar code to below (input x and input y should still be the same)
            #then compute loss using same thing
            #add loss
            #return graph

            #we need our code to approx sin

            # print("y type " + str(type(y)))
            input_y = nn.Input(self.graph, y)
            loss = nn.SquareLoss(self.graph, xm_plus_b2, input_y) # adds loss node to graph
            return self.graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(xm_plus_b2)


class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.02 # adjust as necessary

        ## Layer 1
        d = 100 # end value of this layer
        self.m = nn.Variable(1, d)
        self.b = nn.Variable(d)

        ## Layer 2
        d2 = 1 # end value of last layer must always be 1
        self.m2 = nn.Variable(d, d2)
        self.b2 = nn.Variable(d2)


    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        ### Odd Regression Model

        self.graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_x = nn.Input(self.graph, x)

        ### Construct g(x)

        ## Layer 1
        xm = nn.MatrixMultiply(self.graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(self.graph, xm, self.b)
        xm_plus_b_relu = nn.ReLU(self.graph, xm_plus_b)

        ## Layer 2
        xm2 = nn.MatrixMultiply(self.graph, xm_plus_b_relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(self.graph, xm2, self.b2) ## construction of g(x)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2) 


        ### Construct -g(-x)
        input_neg_one = nn.Input(self.graph, np.array([[-1.0]]))
        neg_x = nn.MatrixMultiply(self.graph, input_x, input_neg_one)
        
        ## Layer 1
        neg_xm = nn.MatrixMultiply(self.graph, neg_x, self.m)
        neg_xm_plus_b = nn.MatrixVectorAdd(self.graph, neg_xm, self.b)
        neg_xm_plus_b_relu = nn.ReLU(self.graph, neg_xm_plus_b)

        ## Layer 2
        neg_xm2 = nn.MatrixMultiply(self.graph, neg_xm_plus_b_relu, self.m2)
        neg_xm_plus_b2 = nn.MatrixVectorAdd(self.graph, neg_xm2, self.b2) ## construction of -g(-x)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2) 

        ## multipy g(-x) * -1
        neg_output = nn.MatrixMultiply(self.graph, neg_xm_plus_b2, input_neg_one)


        ### Add g(x) + (-g(x))
        predicted_y = nn.Add(self.graph, xm_plus_b2, neg_output)


        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SquareLoss(self.graph, predicted_y, input_y) # adds loss node to graph
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(predicted_y)


class DigitClassificationModel(Model):
    """
    TODO: Question 6 - [Application] Digit Classification

    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.15 # adjust as necessary

        ## Layer 1
        d = 350 # end value of this layer, 200 or 250
        self.m = nn.Variable(784, d)
        self.b = nn.Variable(d)

        ## Layer 2
        d2 = 10 # end value of last layer must always be 1
        self.m2 = nn.Variable(d, d2)
        self.b2 = nn.Variable(d2)

    def run(self, x, y=None):
        """
        TODO: Question 6 - [Application] Digit Classification

        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        ### Digit Classification 

        self.graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_x = nn.Input(self.graph, x)

        ## Layer 1
        xm = nn.MatrixMultiply(self.graph, input_x, self.m)
        xm_plus_b = nn.MatrixVectorAdd(self.graph, xm, self.b)
        xm_plus_b_relu = nn.ReLU(self.graph, xm_plus_b)

        ## Layer 2
        xm2 = nn.MatrixMultiply(self.graph, xm_plus_b_relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(self.graph, xm2, self.b2) ## construction of g(x)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2) 

        predicted_y = xm_plus_b2

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(self.graph, y)
            loss = nn.SoftmaxLoss(self.graph, predicted_y, input_y) # adds loss node to graph
            return self.graph
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(predicted_y)


class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.007 # adjust as necessary

        ## Layer 1
        d = 350 # end value of this layer, 200 or 250
        self.m = nn.Variable(4, d)
        self.b = nn.Variable(d)

        ## Layer 2
        d2 = 2 # end value of last layer must always be 1
        self.m2 = nn.Variable(d, d2)
        self.b2 = nn.Variable(d2)


    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        ### Deep Q-Learning

        self.graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_states = nn.Input(self.graph, states)

        ## Layer 1
        xm = nn.MatrixMultiply(self.graph, input_states, self.m)
        xm_plus_b = nn.MatrixVectorAdd(self.graph, xm, self.b)
        xm_plus_b_relu = nn.ReLU(self.graph, xm_plus_b)

        ## Layer 2
        xm2 = nn.MatrixMultiply(self.graph, xm_plus_b_relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(self.graph, xm2, self.b2) ## construction of g(x)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2) 

        predicted_y = xm_plus_b2

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_q_target = nn.Input(self.graph, Q_target)
            loss = nn.SquareLoss(self.graph, predicted_y, input_q_target) # adds loss node to graph
            return self.graph
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(predicted_y)


    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.007 # adjust as necessary

        ## Layer 1
        d = 350 # end value of this layer, 200 or 250
        self.m = nn.Variable(4, d)
        self.b = nn.Variable(d)

        ## Layer 2
        d2 = 2 # end value of last layer must always be 1
        self.m2 = nn.Variable(d, d2)
        self.b2 = nn.Variable(d2)

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        ### Language ID Model

        """
        1. one hot encode each character C into a vector C_0 --> ascii ( d = 8)
        2. h_0 starts out as a zero vector of dimensionality d
        3. h1 = f(h0, c0) using the xm + b, relu format
        4. ... hn = f(hn, cn) where n is the number of characters in the input word (for loop)
        row of matrix h

        At the end
        5. pass final h into one more round of neural network and output predicted y
        """

        self.graph = nn.Graph([self.m, self.b, self.m2, self.b2])
        input_xs = nn.Input(self.graph, xs)
        h = np.zeros_like((5,1))

        for c in range(len(xs)): # c is the ith letter of n words
            # Encode c
            for i in range(num_chars):
                # np.transpose

            # Layer 
            xm = nn.MatrixMultiply(self.graph, input_xs, self.m)
            xm_plus_b = nn.MatrixVectorAdd(self.graph, xm, self.b)
            xm_plus_b_relu = nn.ReLU(self.graph, xm_plus_b)

        # Last layer
        xm2 = nn.MatrixMultiply(self.graph, xm_plus_b_relu, self.m2)
        xm_plus_b2 = nn.MatrixVectorAdd(self.graph, xm2, self.b2) ## construction of g(x)
        # xm_plus_b_relu2 = nn.ReLU(self.graph, xm_plus_b2) 

        predicted_y = xm_plus_b2
        if y is not None:
            "*** YOUR CODE HERE ***"
            input_q_target = nn.Input(self.graph, Q_target)
            loss = nn.SoftmaxLoss(self.graph, predicted_y, input_q_target) # adds loss node to graph
            return self.graph
        else:
            "*** YOUR CODE HERE ***"
            return self.graph.get_output(predicted_y)
