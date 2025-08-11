# ml-math

`ml-math` is a **Rust-powered Python library** that delivers **high-speed mathematical, statistical, and machine learning functions**.  
It is built with [PyO3](https://pyo3.rs/) and packaged with [maturin](https://github.com/PyO3/maturin), meaning you get **native Rust performance** with the simplicity of Python.

---

## What is `ml-math`?

`ml-math` (short for **Machine Learning Mathematics**) is a single Python package that provides a wide set of **numerical and machine learning helper functions** commonly needed in:

- Machine learning preprocessing
- Statistical analysis
- Scientific computing
- Data normalization
- Loss function calculations
- Neural network activations

Instead of relying on slow Python loops or multiple packages, `ml-math` implements these functions **directly in Rust** and exposes them to Python for **maximum speed and efficiency**.

---

## Why Rust + Python?

Python is the dominant language for data science, but heavy numerical computations in Python alone can be **slow** without NumPy or C extensions.  
Rust offers:

- **Memory safety** without garbage collection
- **Zero-cost abstractions**
- **Fast native performance**
- **Cross-platform compatibility**

With `ml-math`, you get **Rust speed** with **Python usability**.

---

## Features

`ml-math` currently includes:

### **Statistics & Mathematics**
- `mean(data)` – Average of values  
- `variance(data)` – Statistical variance  
- `std_dev(data)` – Standard deviation  
- `rms(data)` – Root mean square  

### **Vector Operations**
- `dot(a, b)` – Dot product of two vectors  
- `euclidean(a, b)` – Euclidean distance  
- `cosine_similarity(a, b)` – Cosine similarity between two vectors  

### **Activation Functions**
- `sigmoid(x)` – Sigmoid function  
- `relu(x)` – Rectified Linear Unit  
- `leaky_relu(x, alpha)` – Leaky ReLU variant  
- `tanh_act(x)` – Hyperbolic tangent function  

### **Loss Functions**
- `mse(y_pred, y_true)` – Mean Squared Error  
- `cross_entropy(y_pred, y_true)` – Cross-entropy loss  
- `log_loss(y_pred, y_true)` – Logarithmic loss  

### **Normalization & Scaling**
- `min_max_normalize(data)` – Rescales to `[0, 1]` range  
- `z_score_normalize(data)` – Standard score normalization  
- `clamp(x, min, max)` – Restricts a value to a range  

### **Utilities**
- `ema(data, alpha)` – Exponential Moving Average  

---

## Installation

```bash
pip install ml-math


import ml_math as ml

# ===============================
#  Statistics
# ===============================
print("Mean:", ml.mean([1, 2, 3]))            # 2.0
print("Variance:", ml.variance([1, 2, 3]))    # 0.666...
print("Std Dev:", ml.std_dev([1, 2, 3]))      # 0.816...
print("RMS:", ml.rms([1, 2, 3]))              # 2.160...

# ===============================
#  Vector Operations
# ===============================
print("Dot Product:", ml.dot([1, 2], [3, 4]))             # 11
print("Euclidean Distance:", ml.euclidean([1, 2], [4, 6])) # 5.0
print("Cosine Similarity:", ml.cosine_similarity([1, 0], [0, 1]))  # 0.0

# ===============================
#  Activation Functions
# ===============================
print("Sigmoid(1.0):", ml.sigmoid(1.0))       # ~0.731
print("ReLU(-2.0):", ml.relu(-2.0))           # 0.0
print("Leaky ReLU(-2.0, 0.1):", ml.leaky_relu(-2.0, 0.1)) # -0.2
print("Tanh(1.0):", ml.tanh_act(1.0))         # ~0.761

# ===============================
#  Loss Functions
# ===============================
print("MSE:", ml.mse([1, 2], [1, 3]))                  # 0.5
print("Cross Entropy:", ml.cross_entropy([0.8, 0.2], [1, 0])) # ~0.223
print("Log Loss:", ml.log_loss([0.9, 0.1], [1, 0]))    # ~0.105

# ===============================
#  Normalization
# ===============================
print("Min-Max Normalize:", ml.min_max_normalize([1, 2, 3]))  # [0.0, 0.5, 1.0]
print("Z-Score Normalize:", ml.z_score_normalize([1, 2, 3]))  # [-1.224, 0.0, 1.224]
print("Clamp(5, 0, 3):", ml.clamp(5, 0, 3))                   # 3

# ===============================
#  Utilities
# ===============================
print("EMA:", ml.ema([1, 2, 3, 4, 5], 0.5))  # Smooth moving average
