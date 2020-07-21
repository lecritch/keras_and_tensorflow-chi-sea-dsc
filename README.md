
# Tensorflow and Keras

## Modeling

Let's review some modeling concepts we've used to date with [this quick exercise](https://forms.gle/yrPxUp2Xj4R9FeyEA)


We do this to remind ourselves that the basic components of good modeling practice, and even the methods themselves, are _the same_ with Neural Nets as that are with _sklearn_ or _statsmodels_.

The above exercise uses only one train-test split, but is still usefule.  We will be using train, validation, test in this notebook, for good practice.

## Objectives:
- Compare pros and cons of Keras vs TensorFlow
- hands on practice coding a neural network

Wait a second, what is that warning? 
`Using TensorFlow backend.`

<img align =left src="img/keras.png"><br>
### Keras is an API

Coded in Python, that can be layered on top of many different back-end processing systems.

![kerasback](img/keras_tf_theano.png)

While each of these systems has their own coding methods, Keras abstracts from that in streamlined pythonic manner we are used to seeing in other python modeling libraries.

Keras development is backed primarily by Google, and the Keras API comes packaged in TensorFlow as tf.keras. Additionally, Microsoft maintains the CNTK Keras backend. Amazon AWS is maintaining the Keras fork with MXNet support. Other contributing companies include NVIDIA, Uber, and Apple (with CoreML).

## Wait, what's TensorFlow?


## Let's start with tensors

## Tensors are multidimensional matricies

![tensor](img/tensors.png)

### TensorFlow manages the flow of matrix math

That makes neural network processing possible.

![cat](img/cat-tensors.gif)

For our numbers dataset, our tensors from the sklearn dataset were originally tensors of the shape 8x8, i.e.64 pictures.  Remember, that was with black and white images.

For image processing, we are often dealing with color.

What do the dimensions of our image above represent?

Even with tensors of higher **rank**

A matrix with rows and columns only, like the black and white numbers, are **rank 2**.

A matrix with a third dimension, like the color pictures above, are **rank 3**.

When we flatten an image by stacking the rows in a column, we are decreasing the rank. 

When we unrow a column, we increase its rank.

### Wait, what tool am I even using, what's Keras?
## More levers and buttons

Coding directly in **Tensorflow** allows you to tweak more parameters to optimize performance. The **Keras** wrapper makes the code more accessible for developers prototyping models.

![levers](img/levers.jpeg)

### Keras, an API with an intentional UX

- Deliberately design end-to-end user workflows
- Reduce cognitive load for your users
- Provide helpful feedback to your users

[full article here](https://blog.keras.io/user-experience-design-for-apis.html)<br>
[full list of why to use Keras](https://keras.io/why-use-keras/)

### A few comparisons

While you **can leverage both**, here are a few comparisons.

| Comparison | Keras | Tensorflow|
|------------|-------|-----------|
| **Level of API** | high-level API | High and low-level APIs |
| **Speed** |  can *seem* slower |  is a bit faster |
| **Language architecture** | simple architecture, more readable and concise | straight tensorflow is a bit mroe complex |
| **Debugging** | less frequent need to debug | difficult to debug |
| **Datasets** | usually used for small datasets | high performance models and large datasets that require fast execution|

This is also a _**non-issue**_ - as you can leverage tensorflow commands within keras and vice versa. If Keras ever seems slower, it's because the developer's time is more expensive than the GPUs. Keras is designed with the developer in mind. 


[reference link](https://www.edureka.co/blog/keras-vs-tensorflow-vs-pytorch/)

### Think, Pair, Share Challenge:

<img src="https://images.pexels.com/photos/1350560/pexels-photo-1350560.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" alt="diabetes" style ="text-align:center;width:250px;float:none" ></br>

Let's begin implementing our neural net with the UCI digit dataset we imported from sklearn yesterday.

Let's continue where we left off with our numbers dataset.

We will start with a binary classification, and predict whether the number will be even or odd.

In pairs, proceed through the following three parts. 

#### Part 1:
Questions to answer:
- How many input variables are there in this dataset? 
- What does the range of values (0-16) represent in our feature set?
- What does a 1 mean in our target class?
- If we use a neural net to predict this, what loss function do we use?
***

***
#### Part 2:
What if you wanted to create a NN with hidden layers to predict even numbers with:
- 12 nodes in the first hidden layer
- 8 nodes in the second hidden layer
- relu on the first two activations
- sigmoid on the last one

Answer the following questions:
- How many nodes in the input layer?
- How many nodes in the output layer?
- Will the output layer produce an integer or a float?
***

***

#### Part 3:
Knowing that you want:
- batch size of 10
- 50 epochs
- to use `rmsprop` as your optimizer
- and all the numbers you defined above...

**Fill out the code below with the correct specifications, but don't run it yet**


```python

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=64,))
model.add(Dense(8 ,  activation='relu' ))
model.add(Dense(1 , activation = 'sigmoid' ))

model.compile(optimizer='rmsprop' ,
              loss='binary_crossentropy'  ,
              metrics=['accuracy'])
model.fit(X, y_binary, epochs=50, batch_size= 10 )
```

    Epoch 1/50
    1797/1797 [==============================] - 2s 1ms/step - loss: 0.5585 - acc: 0.7446
    Epoch 2/50
    1797/1797 [==============================] - 0s 203us/step - loss: 0.2448 - acc: 0.9009
    Epoch 3/50
    1797/1797 [==============================] - 0s 198us/step - loss: 0.1640 - acc: 0.9349
    Epoch 4/50
    1797/1797 [==============================] - 0s 177us/step - loss: 0.1305 - acc: 0.9482
    Epoch 5/50
    1797/1797 [==============================] - 0s 177us/step - loss: 0.1068 - acc: 0.9638
    Epoch 6/50
    1797/1797 [==============================] - 0s 167us/step - loss: 0.0902 - acc: 0.9705
    Epoch 7/50
    1797/1797 [==============================] - 0s 159us/step - loss: 0.0793 - acc: 0.9716
    Epoch 8/50
    1797/1797 [==============================] - 0s 190us/step - loss: 0.0715 - acc: 0.9766
    Epoch 9/50
    1797/1797 [==============================] - 0s 179us/step - loss: 0.0621 - acc: 0.9789
    Epoch 10/50
    1797/1797 [==============================] - 0s 160us/step - loss: 0.0564 - acc: 0.9794
    Epoch 11/50
    1797/1797 [==============================] - 0s 157us/step - loss: 0.0516 - acc: 0.9822
    Epoch 12/50
    1797/1797 [==============================] - 0s 196us/step - loss: 0.0475 - acc: 0.9822
    Epoch 13/50
    1797/1797 [==============================] - 0s 168us/step - loss: 0.0417 - acc: 0.9861
    Epoch 14/50
    1797/1797 [==============================] - 0s 168us/step - loss: 0.0402 - acc: 0.9855
    Epoch 15/50
    1797/1797 [==============================] - 0s 203us/step - loss: 0.0351 - acc: 0.9894
    Epoch 16/50
    1797/1797 [==============================] - 0s 164us/step - loss: 0.0324 - acc: 0.9878
    Epoch 17/50
    1797/1797 [==============================] - 0s 175us/step - loss: 0.0292 - acc: 0.9905
    Epoch 18/50
    1797/1797 [==============================] - 0s 181us/step - loss: 0.0276 - acc: 0.9894
    Epoch 19/50
    1797/1797 [==============================] - 0s 164us/step - loss: 0.0248 - acc: 0.9905
    Epoch 20/50
    1797/1797 [==============================] - 0s 186us/step - loss: 0.0239 - acc: 0.9939
    Epoch 21/50
    1797/1797 [==============================] - 0s 179us/step - loss: 0.0211 - acc: 0.9933
    Epoch 22/50
    1797/1797 [==============================] - 0s 180us/step - loss: 0.0206 - acc: 0.9922
    Epoch 23/50
    1797/1797 [==============================] - 0s 164us/step - loss: 0.0193 - acc: 0.9922
    Epoch 24/50
    1797/1797 [==============================] - 0s 175us/step - loss: 0.0188 - acc: 0.9939
    Epoch 25/50
    1797/1797 [==============================] - 0s 172us/step - loss: 0.0142 - acc: 0.9972
    Epoch 26/50
    1797/1797 [==============================] - 0s 164us/step - loss: 0.0139 - acc: 0.9950
    Epoch 27/50
    1797/1797 [==============================] - 0s 160us/step - loss: 0.0117 - acc: 0.9967
    Epoch 28/50
    1797/1797 [==============================] - 0s 190us/step - loss: 0.0116 - acc: 0.9961
    Epoch 29/50
    1797/1797 [==============================] - 0s 186us/step - loss: 0.0106 - acc: 0.9950
    Epoch 30/50
    1797/1797 [==============================] - 0s 187us/step - loss: 0.0092 - acc: 0.9972
    Epoch 31/50
    1797/1797 [==============================] - 0s 193us/step - loss: 0.0120 - acc: 0.9955
    Epoch 32/50
    1797/1797 [==============================] - 0s 206us/step - loss: 0.0094 - acc: 0.9967
    Epoch 33/50
    1797/1797 [==============================] - 0s 218us/step - loss: 0.0082 - acc: 0.9983
    Epoch 34/50
    1797/1797 [==============================] - 0s 189us/step - loss: 0.0058 - acc: 0.9978
    Epoch 35/50
    1797/1797 [==============================] - 0s 176us/step - loss: 0.0081 - acc: 0.9972
    Epoch 36/50
    1797/1797 [==============================] - 0s 195us/step - loss: 0.0067 - acc: 0.9972
    Epoch 37/50
    1797/1797 [==============================] - 0s 181us/step - loss: 0.0066 - acc: 0.9983
    Epoch 38/50
    1797/1797 [==============================] - 0s 187us/step - loss: 0.0052 - acc: 0.9983
    Epoch 39/50
    1797/1797 [==============================] - 0s 203us/step - loss: 0.0035 - acc: 0.9989
    Epoch 40/50
    1797/1797 [==============================] - 0s 181us/step - loss: 0.0064 - acc: 0.9978
    Epoch 41/50
    1797/1797 [==============================] - 0s 178us/step - loss: 0.0043 - acc: 0.9972
    Epoch 42/50
    1797/1797 [==============================] - 0s 202us/step - loss: 0.0041 - acc: 0.9989
    Epoch 43/50
    1797/1797 [==============================] - 0s 199us/step - loss: 0.0037 - acc: 0.9994
    Epoch 44/50
    1797/1797 [==============================] - 0s 194us/step - loss: 0.0044 - acc: 0.9989
    Epoch 45/50
    1797/1797 [==============================] - 0s 222us/step - loss: 0.0041 - acc: 0.9989
    Epoch 46/50
    1797/1797 [==============================] - 0s 185us/step - loss: 0.0056 - acc: 0.9983
    Epoch 47/50
    1797/1797 [==============================] - 0s 191us/step - loss: 0.0027 - acc: 0.9994
    Epoch 48/50
    1797/1797 [==============================] - 0s 201us/step - loss: 0.0031 - acc: 0.9994
    Epoch 49/50
    1797/1797 [==============================] - 0s 167us/step - loss: 0.0023 - acc: 0.9994
    Epoch 50/50
    1797/1797 [==============================] - 0s 173us/step - loss: 0.0034 - acc: 0.9994





    <keras.callbacks.History at 0x1a60c6cda0>



### Things to know:
- the data and labels in `fit()` need to be numpy arrays, not pandas dfs. Else it won't work.
- Scaling your data will have a large impact on your model.   
  > For our traditional input features, we would use a scalar object.  For images, as long as the minimum value is 0, we can simply divide through by the maximum pixel intensity.

![gif](https://media0.giphy.com/media/3og0IMJcSI8p6hYQXS/giphy.gif)

#### Getting data ready for modeling
**Tasks**:

- use train_test_split to create X_train, y_train, X_test, and y_test
- Split training data into train and validation sets.
- Scale the pixel intensity to a value between 0 and 1.
- Scale the pixel intensity to a value between 0 and 1.

Scaling data for neural networks is very important, whether it be for image processing or prediction problems like we've seen in past projects and lessons.  

Scaling our input variables will help speed up our neural network [see 4.3](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

Since our minimum intensity is 0, we can normalize the inputs by dividing each value by the max value (16). 


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, random_state=42, test_size=.2)
X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, random_state=42, test_size=.2)
X_t, X_val, X_test = X_t/16, X_val/16, X_test/16

```

Now that our data is ready, let's load in keras

Let's start working through the different choices we can make in our network.

For activation, let's start with the familiar sigmoid function, and see how it performs.

If we look at our loss, it is still decreasing. That is a signal that our model is still learning. If our model is still learning, we can allow it to get better by increasing the number of epochs.

It still looks like our model has not converged.  The loss is still decreasing, and the accuracy is still increasing.  We could continue increasing the epochs, but that will be time consuming.  

We could try decreasing the batch size. Let's set the batch size to 1.  This is true stochastic gradient descent.  The parameters are updated after each sample is passed into the model.

SGD with a small batch size takes longer to run through an epoch, but will take less epochs to improve.

Comparing our 50 epoch version with a 500 batch size and a 10 epoch version with a 1 example batch size, we see that by 10 epochs, the latter has achieved 90% accuracy by the final epoch, while our 23 batch size is just about 70%.  However, with the 1 example batch, each epoch took a lot longer.

Still, even though the 2nd model reached a higher accuracy and lower loss, it looks like it still has not stopped learning. The slope of the loss is getting smaller, but it has not leveled out completely.

Let's get a bit more modern, and apply a relu activation function in our layers.

We are reaching a high accuracy, but still looks like our model has not converged. If we increased our number of epochs, we would be looking at a long wait.

We have been implementing the vanilla version of gradient descent.  Remember, SGD updates the parameters uniformly across the board.  Let's try out an optimizer used more often these days.

Now our accuracy is really improving, and it looks like our learning may be leveling out.

Since Adam and relu are relatively faster than SGD and sigmoid, we can add more epochs, and more layers without the training time getting unwieldy.

No it looks like we're getting somewhere.

For comparison, look at how much more quickly Adam learns than SGD.

We have been looking only at our training set. Let's add in our validation set to the picture.

Consider that we still see our loss decreasing and our accuracy increasing.  We try to add more complexity to our model by adding more layers.

We see that our model is overfit.  Just like in our previous models, after a certain amount of learning, the loss on the validation set starts increasing.

# Regularization


Does regularization make sense in the context of neural networks? <br/>

Yes! We still have all of the salient ingredients: a loss function, overfitting vs. underfitting, and coefficients (weights) that could get too large.

But there are now a few different flavors besides L1 and L2 regularization. (Note that L1 regularization is not common in the context of  neural networks.)

# Early Stopping

Now let's return to the original problem: predicting 0 through 9

Keras comes with all sorts of handy tools, including ways to streamline train test split from folders on your desktop. You will definitely find this useful. Learn will lead the way.

You don't have this dog vs. cat dataset on your computer, but that is ok. 

The code below shows how we process that data with Keras. It also shows that a basic neural net does not perform well on the dataset.  Tomorrow, we will explore better tools for image processing.
