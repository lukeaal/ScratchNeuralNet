import numpy as np
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt
from MLP import MLP  # Ensure MLP is imported correctly

# Set a random seed for reproducibility
rng = default_rng(8675309)

# Read the data
df = pd.read_csv('./data/wine.csv', header=None)

# Separate targets and inputs, randomly shuffle the data
ds = df.sample(frac=1)
wineData = ds.to_numpy()
inputs = wineData[:, 1:]
preTargets = wineData[:, 0]

# Normalize the data
maxes = np.amax(inputs, axis=0)
mins = np.amin(inputs, axis=0)

# Normalize the data
for i in range(len(inputs)):
    for j in range(len(mins)):
        inputs[i][j] = (maxes[j] - inputs[i][j]) / (maxes[j] - mins[j])

# Change the end class into three columns, values of 1 or 0 (one-hot encoded)
targets = []
for i in preTargets:
    if i == 1:
        targets.append([1, 0, 0])
    if i == 2:
        targets.append([0, 1, 0])
    if i == 3:
        targets.append([0, 0, 1])

# Targets are now the one-hot encoded version of the three wine classes

# Chunk the data now for 5-fold cross-validation
chunks = []
# Five chunks
chunks.append([inputs[0:35], targets[0:35]])
chunks.append([inputs[35:70], targets[35:70]])
chunks.append([inputs[70:105], targets[70:105]])
chunks.append([inputs[105:140], targets[105:140]])
chunks.append([inputs[140:], targets[140:]])
targets = np.array(targets)

# Join the two partitions of the data back together and separate them into five batches.
# Perform 5-fold cross-validation, report the performance after each epoch
error = 10000000
multi = MLP(inputs[0:130, :], targets[0:130, :], 6, 0.5)
multiTwo = MLP(inputs[35:160, :], targets[35:160, :], 6, 0.5)
mutliThree = MLP(inputs[60:178, :], targets[60:178, :], 6, 0.5)
mutliFour = MLP(inputs[35:170, :], targets[35:170, :], 6, 0.5)
multiFive = MLP(inputs[35:170, :], targets[35:170, :], 6, 0.5)
testData = []
sses = []

# Test one, record accuracy
while error > 0.1:
    multi.train()
    # Multi will test its accuracy on itself until it fits the data 90 percent
    error = multi.sse()
    sses.append(error)

print("Accuracy:", multi.accuracy(inputs[0:35, :], targets[0:35, :]))

# Create a plot for the SSE (Sum of Squared Errors)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sses)
ax.set_xlabel("Epoch")
ax.set_ylabel("SSE")
ax.set_title("SSE vs. Epoch")
plt.show()
