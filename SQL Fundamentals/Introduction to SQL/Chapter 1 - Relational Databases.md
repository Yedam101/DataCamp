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

