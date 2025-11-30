# MNIST Digit Recognizer: Built from Scratch

### 🧠 Pure NumPy Implementation (No TensorFlow/PyTorch)

## 📌 Overview
This project is a raw implementation of a Feed-Forward Neural Network to classify handwritten digits from the MNIST dataset.

Unlike standard implementations that rely on high-level frameworks like TensorFlow or PyTorch, **this project builds the neural network entirely from scratch using only NumPy and Pandas.** The goal was to deconstruct the "black box" of Deep Learning and understand the linear algebra and calculus governing forward propagation and gradient descent.

## 🚀 Key Features
* **Zero ML Frameworks:** Logic is implemented using raw matrix multiplication and calculus.
* **Custom Backpropagation:** Manual implementation of the Chain Rule to calculate gradients.
* **Vectorized Operations:** Uses NumPy broadcasting for efficient computation over batches.
* **Visual Validation:** Includes a prediction visualizer to test the model against specific images.

## 📐 Architecture
The network consists of a 2-layer architecture (Input $\rightarrow$ Hidden $\rightarrow$ Output):

* **Input Layer:** $784$ units (corresponding to $28 \times 28$ pixel images).
* **Hidden Layer:** $10$ neurons with **ReLU** activation.
* **Output Layer:** $10$ neurons with **Softmax** activation (representing probabilities for digits 0-9).

## 🧮 Mathematical Foundation
The core logic relies on the following signal flow designed from first principles:

### 1. Forward Propagation
$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = \text{softmax}(Z^{[2]})$$

### 2. Backpropagation (Derivatives)
The model learns by minimizing the loss via Gradient Descent, calculating derivatives recursively:
$$dZ^{[2]} = A^{[2]} - Y$$
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \times g'(Z^{[1]})$$
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

*(Where m is the number of training examples)*

## 📂 Project Structure
```bash
├── train.csv          # MNIST Training data (pixel values and labels)
├── main.py            # Complete source code (Init, Prop, Descent)
└── README.md          # Project documentation
```

🛠️ Requirements

   * Python 3.x

   * NumPy

   * Pandas

   * Matplotlib (for visualizing predictions)

To install dependencies:
```bash
pip install numpy pandas matplotlib
```

📊 Results

After training for 500 iterations with a learning rate (α) of 0.1:

   * Final Accuracy: ~85%

   * The model successfully converges and is able to predict handwritten digits with high confidence.

🧠 Learning Outcomes

Building this project helped solidify concepts in:

  *  Linear Algebra: Matrix dimensionality and dot products in Neural Networks.

  *  Calculus: Practical application of the Chain Rule for backpropagation.

  *  Optimization: Understanding how Gradient Descent minimizes the Loss function.
