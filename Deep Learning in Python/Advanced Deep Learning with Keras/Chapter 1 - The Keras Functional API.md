## chapter 1-1

Input layers

```python
# Import Input from keras.layers

from keras.layers import Input

# Create an input layer of shape 1
input_tensor = Input(shape=(1,))

```

## chapter 1-2

Dense layers

```python
# Load layers
from tensorflow.keras.layers import Input, Dense


# Input layer
input_tensor = Input(shape=(1,))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)

```

## chapter 1-3

Output layers

```python
# Load layers
from keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)

```

## chapter 1-4

Build a model

```python
# Input/dense/output layers
from tensorflow.keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)

# Build the model
from tensorflow.keras.models import Model 
model = Model(input_tensor, output_tensor)

```

## chapter 1-5

Compile a model

```python
# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

```

## chapter 1-6

Visualize a model

```python
# Import the plotting function
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()

```

## chapter 1-7

Fit the model to the tournament basketball data

```python
# Train your model for 20 epochs
# Now fit the model
model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs=1,
          batch_size=128,
          validation_split=0.1,
          verbose=True)

```

## chapter 1-8

Evaluate the model on a test set

```python
# Load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# Load the y variable from the test data
y_test = games_tourney_test['score_diff']

# Evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose=False))

```

