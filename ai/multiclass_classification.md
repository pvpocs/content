# Multiclass Classification
Multiclass classification is a classification task that the target variable can take on more than two values. In other words, the target variable $y$ can take on $N$ different classes, where $N > 2$.

Some of the examples of multiclass classification problems include:

| Question | Target Variable | Classes |
| --- | --- | --- |
| What type of tumor is this? | Malignant, Benign, Normal | $N=3$ |
| Hand written digit recognition | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | $N=10$ |
| What type of animal is this? | Cat, Dog, Bird | $N=3$ |
| What type of vehicle is this? | Car, Truck, Bus, Motorcycle, Bicycle | $N=5$ |
| Next word (English) prediction in NLP | [Vocabulary of words/tokens](https://arxiv.org/abs/2406.16508) | $N=50,000+$ |


In binary classification, the target variable $y$ can take only two values, 0 or 1. That's why an algorithm like logistic regression which uses the sigmoid function is an appropriate choice for binary classification.

However, for multiclass classification, where $y$ can take on multiple values, we need a different algorithm such as **Softmax Regression** which is the generalization of logistic regression algorithm.

## Softmax
Softmax regression is a generalization of logistic regression to multiple classes.


**In Binary Classification:**<br>

Using the sigmoid function:
$$z= \vec{\mathbf{w}}.\vec{\mathbf{x}} + b$$

$$P(y=1|\vec{\mathbf{x}};\vec{\mathbf{w}},b) = \frac{1}{1 + e^{-z}}$$

The output of the model is probabilities of the target variable $y$ being 1 or 0.

The total probability of the target variable being 1 or 0 should be 1, so:

$$P(y=1|\vec{\mathbf{x}};\vec{\mathbf{w}},b) + P(y=0|\vec{\mathbf{x}};\vec{\mathbf{w}},b) = 1$$



**MultiClass Classification:**<br>
The output is a vector of $N$ values, where $N$ is the number of classes. The output vector is then passed through the softmax function to get the probabilities of each class.

Total probability of all classes should be 1.
$$P(y=1|x;\vec{\mathbf{w}},b) + P(y=2|x;\vec{\mathbf{w}},b) + ... + P(y=N|x;\vec{\mathbf{w}},b) = 1$$

Which can be written as:
$$\sum_{i=1}^{N} P(y=i|x;\vec{\mathbf{w}},b) = 1$$
