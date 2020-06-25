import nn
import numpy as np
import math
import random

class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new LogisticRegressionModel instance.

        A logistic regressor classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        Initialize self.w and self.alpha here
        self.alpha = *some number* (optional)
        self.w = []
        """

        "*** YOUR CODE HERE ***"
        self.w = []
        self.alpha = 0.009
        # self.w = nn.Parameter(1, dimensions)
        for i in range(0,dimensions):
            self.w.append(random.random())
    def get_weights(self):
        """
        Return a list of weights with the current weights of the regression.
        """
        return self.w

    def DotProduct(self, w, x):
        """
        Computes the dot product of two lists
        Returns a single number
        """
        "*** YOUR CODE HERE ***"
        return np.dot(self.w, x)

    def sigmoid(self, x):
        """
        compute the logistic function of the input x (some number)
        returns a single number
        """
        "*** YOUR CODE HERE ***"
        # dot_pdt = self.DotProduct(self.w,x)
        z = 1./(1 + math.exp(-x))
        return z

    def run(self, x):
        """
        Calculates the probability assigned by the logistic regression to a data point x.

        Inputs:
            x: a list with shape (1 x dimensions)
        Returns: a single number (the probability)
        """
        "*** YOUR CODE HERE ***"
        # print(dot_pdt)
        wTx = self.DotProduct(self.w, x)
        act_func = self.sigmoid(wTx)
        return act_func


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if self.run(x) >= 0.5:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the logistic regression until convergence (this will require at least two loops).
        Use the following internal loop stucture

        for x,y in dataset.iterate_once(1):
            x = nn.as_vector(x)
            y = nn.as_scalar(y)
            ...

        """
        "*** YOUR CODE HERE ***"
        loss = -1.0
        while True:
            prev_loss = loss
            loss = 0.0
            for x, y in dataset.iterate_once(1):
                x = nn.as_vector(x)
                y = nn.as_scalar(y)
                if y!=self.get_prediction(x):
                    for ind in range(1,len(x)):
                        self.w[ind] = self.w[ind] + self.alpha * (y-self.run(x))*self.run(x)*(1-self.run(x))*x[ind]
                    loss = loss + (y - self.run(x)) ** 2
            if prev_loss == loss:
                    break

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        wTx = nn.DotProduct(self.w, x)
        # print(type(wTx))
        return wTx

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        if nn.as_scalar(self.run(x)) >= 0.0 :
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        check = True
        while check:
            for x,y in dataset.iterate_once(1):
                y = nn.as_scalar(y)
                if y != self.get_prediction(x):
                    check = False
                    nn.Parameter.update(self.w, x, y)

            if not check:
                check = True
            else:
                break



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, dimensions):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w = nn.Parameter(1, dimensions)
        self.bias = nn.Parameter(1, 1)
        self.alpha = 0.009

        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        wTx = nn.Linear(x, self.w)
        return nn.AddBias(wTx, self.bias)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        z = self.run(x)
        return nn.SquareLoss(z, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            loss = 0
            for x,y in dataset.iterate_once(50):
                gradients = nn.gradients(self.get_loss(x,y), [self.w,self.bias])
                self.w.update(gradients[0],-self.alpha)
                self.bias.update(gradients[1],-self.alpha)
                loss += nn.as_scalar(self.get_loss(x,y))
            print(loss)
            if loss < 0.6:
                break

    def closedFormSolution(self, X, Y):
        """
        Compute the closed form solution for the 2D case
        Input: X,Y are lists
        Output: b0 and b1 where y = b1*x + b0
        """
        "*** YOUR CODE HERE ***"
        sum1 = 0
        for i in range(0,len(X)):
            sum1 += X[i]*Y[i]
        
        sum_2_x = 0
        for x in X:
            sum_2_x += x
        
        sum_3_y = 0
        for y in Y:
            sum_3_y += y

        sum_4_sq = 0
        for x in X:
            sum_4_sq += x**2

        b1 = ((len(X) * (sum1)) - (sum_2_x * sum_3_y))/((len(X) * sum_4_sq) - (sum_2_x**2))
        b0 = (sum_3_y - (b1 * sum_2_x))/len(X)
        print(b1)
        print(b0)
        return b0,b1


class PolyRegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, order):
        # Initialize your model parameters here
        """
        initialize the order of the polynomial, as well as two parameter nodes for weights and bias
        """
        "*** YOUR CODE HERE ***"
        self.degree = order
        self.w = nn.Parameter(order, 1)
        self.b = nn.Parameter(1, 1)  # bias term
        self.alpha = 0.001 #learning rate

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        features = self.computePolyFeatures(x)
        wTx = nn.Linear(features,self.w)
        return nn.AddBias(wTx,self.b)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        z = self.run(x)
        return nn.SquareLoss(z, y)

    def computePolyFeatures(self, point):
        """
        Compute the polynomial features you need from the input x
        NOTE: you will need to unpack the x since it is wrapped in an object
        thus, use the following function call to get the contents of x as a list:
        point_list = nn.as_vector(point)
        Once you do that, create a list of the form (for batch size of n): [[x11, x12, ...], [x21, x22, ...], ..., [xn1, xn2, ...]]
        Once this is done, then use the following code to convert it back into the object
        nn.Constant(nn.list_to_arr(new_point_list))
        Input: a node with shape (batch_size x 1)
        Output: an nn.Constant object with shape (batch_size x n) where n is the number of features generated from point (input)
        """
        "*** YOUR CODE HERE ***"
        point_list = nn.as_vector(point)
        poly_batch_list = []
        for ind in range(0,len(point_list)):
            
            powers_list = []
            for i in range(1,self.degree+1):
                powers_list.append(i)
            
            con_cat_list = []
            for i in range(0, self.degree):
                con_cat_list.append(point_list[ind])

            features = []
            for i in range(0, len(con_cat_list)):
                features.append(np.power(con_cat_list[i], powers_list[i]))
            
            poly_batch_list.append(features)
        
        return nn.Constant(nn.list_to_arr(poly_batch_list))

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            loss = 0
            for x,y in dataset.iterate_once(10):
                gradients = nn.gradients(self.get_loss(x,y), [self.w,self.b])
                self.w.update(gradients[0],-self.alpha)
                self.b.update(gradients[1],-self.alpha)
                # print(type(self.get_loss(x,y)))
                loss += nn.as_scalar(self.get_loss(x,y))
            print(loss)
            if loss < 0.5:
                break

class FashionClassificationModel(object):
    """
    A model for fashion clothing classification using the MNIST dataset.

    Each clothing item is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        return

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        return
