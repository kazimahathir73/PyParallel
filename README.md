# PyParallel

**PyParallel** is a PyTorch-based deep learning framework designed to simplify model and data parallelism for deep learning. It provides powerful tools for model parallel training, data parallelism, and optimization techniques for CPU, GPU, and TPU. Additionally, it offers real-time hardware monitoring and memory tracking, helping users keep track of system resources during training.

Video Tutorial - 

## Features

- **Model Parallelism**: Enables efficient training of large models across multiple devices.
- **Data Parallelism**: Allows splitting of data batches across multiple devices for faster training.
- **Optimization Techniques**:
  - **Gradient Checkpointing**: Reduces memory usage by saving checkpoints during training.
  - **Mixed Precision**: Improves speed and efficiency by using lower precision for training.
  - **Automatic Loss Scaling**: Prevents underflow during mixed-precision training.
  - **Gradient Clipping**: Controls gradient magnitude to stabilize training.
  - **Gradient Accumulation**: Allows training with smaller memory by accumulating gradients over multiple batches.
- **Hardware Monitoring**:
  - Provides system information, including details on available CPU, GPU, and TPU cores.
  - Real-time memory usage tracking for CPU, GPU, and TPU during training sessions.

## Installation

To install **PyParallel**, you can use the following steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kazimahathir73/PyParallel.git
   cd PyParallel
   ```

2. Install the library and its dependencies:
   ```bash
   pip install .
   ```

Alternatively, you can install directly from PyPI (if published):
```bash
pip install pyparallel
```
## Quickstart

To get started quickly with PyParallel, check out the following example demonstrating model training with data parallelism and optimization techniques. This example is also available as a Jupyter notebook for interactive use: [Demo Notebook](https://github.com/kazimahathir73/PyParallel/example/demo_training.ipynb/).

## Documentation

For detailed documentation, and advanced examples, please refer to the [PyParallel Documentation](https://github.com/kazimahathir73/PyParallel/docs/manual.md).