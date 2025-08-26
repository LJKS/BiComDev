import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


import agents
import data


# Both agents A1 and A2 receive a respective set of feature vectors FA1 and
# FA2 sampled from a dataset D and aim to predict the target set, defined as
# the intersection Y = FA1∩FA2, represented by a binary target vector yA1,
# yA2 based on the indicator function2 to an enumerated representation of
# the respective set of feature vectors: yA1 = [1Y (FA11),...,1Y (FA1 |F|)]
# yA2 accordingly. The game size |F| = |FA1| = |FA2| stays fixed through
# all iterations of the game, but the feature vectors FA1 and FA2, and the
# target size |T| vary between iterations.


# load data (images, features, annotations?)

# load agents


# train_step function


# train function