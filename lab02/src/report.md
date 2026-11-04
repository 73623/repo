Lab 2: A Neural Network from Scratch

 1. Goal

The goal of this lab was to design, implement, and evaluate a feedforward neural network with two hidden layers. The network was built using matrix operations. The work included implementing forward propagation, cross-entropy loss, and backpropagation; checking the calculated gradients against numerical estimates; training with different learning rates and hidden-layer widths; and measuring execution times.

 2. Method

 Dataset and preparation

The experiment used the handwritten digits dataset from scikit-learn. It contains 1,797 samples, each represented by 64 pixel values. Pixel values were scaled to the range from zero to one by dividing by 16. The digit labels were represented as ten-class targets.

The data was divided into training and test sets, with 70 percent used for training and 30 percent for testing. The split preserved the proportion of each digit class and used a fixed random seed of 42.

 Network design

The network accepts 64 input features. It passes them through two hidden layers, each containing the same number of units, and then produces scores for the ten possible digits. The hidden layers use the ReLU activation function, and the output layer uses softmax to produce class probabilities.

The first layer connects the 64 input features to the hidden units. The second layer connects the first hidden layer to the second. The final layer connects the second hidden layer to the ten output classes. Each layer has weights and biases. Weights were initialized with small random values from a normal distribution with a standard deviation of 0.01, and biases were initialized to zero.

 Training and loss

For each batch, the network computes activations through the layers and compares its predicted class probabilities with the correct labels. The difference is measured using categorical cross-entropy, averaged across the samples in the batch.

Analytical gradients were calculated using the chain rule. Their correctness was checked against central-difference numerical estimates, using a perturbation of 0.00001. The allowed relative error was less than 0.000001.

 3. Results

Gradient check

The target was for at least 9 of 10 randomly selected weights to pass the gradient check. All 10 checked weights passed, and their relative errors were below 0.0000001.

Initial loss check

Before training, the expected loss for predictions that assign equal probability to all ten digits was approximately 2.3026. The observed initial loss was also 2.3026.

 Hyperparameter results

Each configuration used a mini-batch size of 32 and was trained for 50 epochs.

| Hidden units per layer | Learning rate | Final cross-entropy loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.3012 | 10.19% |
| 16 | 0.01 | 2.2985 | 18.52% |
| 16 | 0.1 | 0.1843 | 92.59% |
| 64 | 0.001 | 2.3019 | 10.19% |
| 64 | 0.01 | 2.2951 | 22.41% |
| 64 | 0.1 | 0.0812 | 96.11% |

Training loss curves were saved to results/loss_curves.png.

 Timing

For a hidden width of 64, a batch size of 32, and 100 benchmark iterations, the median forward-pass time was 0.0841 milliseconds. The median backward-pass time was 0.1812 milliseconds. The backward pass was approximately 2.15 times slower than the forward pass.

4. Conclusions

1. **Learning rate affected convergence.** With the chosen small weight initialization, learning rates of 0.001 and 0.01 made little progress within 50 epochs. A learning rate of 0.1 produced much lower loss and better accuracy. The best test accuracy was 96.11%, achieved with 64 units in each hidden layer.

2. **The wider network performed better in this experiment.** With a learning rate of 0.1, increasing the hidden-layer width from 16 to 64 reduced the final loss from 0.1843 to 0.0812 and improved test accuracy from 92.59% to 96.11%.

3. **Backpropagation took longer than forward propagation.** The backward pass was about 2.15 times slower in the benchmark. It computes gradients through the network, which requires additional matrix operations beyond those used to produce predictions.
