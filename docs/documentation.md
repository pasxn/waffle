<div align="center">

# Waffle Documentation

</div>

## Architecture

When the package is installed, setup script registers the [waffle](https://github.com/pasxn/waffle/tree/main/waffle) core directory as a pip package. The package has 5 submodules.

* [**onnx**](https://github.com/pasxn/waffle/tree/main/waffle/onnx): Contains the code required to read the serialized ONNX graph from the .onnx file and create an in memory construction of the graph to traverse in the inference time.

* [**nn**](https://github.com/pasxn/waffle/tree/main/waffle/nn): Contains the API for the neural network inference, and the implementations of supported operations.

* [**tensor**](https://github.com/pasxn/waffle/tree/main/waffle/tensor.py): A custom made tensor library and its API.

* [**ops**](https://github.com/pasxn/waffle/tree/main/waffle/ops.py): A level of abstraction designed to map each operation to the target accelerator backend (waffle currently only has a CPU backend).

* [**backend**](https://github.com/pasxn/waffle/tree/main/waffle/backend): Backend specific code, compilation routines and the API for each backend.

## Primitive Operations

There are 12 primitive operations defined in waffle which are commonly used when implementing simple neural network architectures. These operations can be used through the APIs to implement custom neural network operations.

* neg  : negative
* exp  : exponential
* log  : logarithm
* relu : rectified linear unit
* add  : addition
* sub  : subtraction
* mul  : multiplication
* div  : division
* pow  : power
* gemm : general matrix multiplication
* sum  : summation
* max  : maximum

## API Documentation

Refer [abstractions.py](https://github.com/pasxn/waffle/blob/main/extra/demo/abstractions.py) and [inference.py](https://github.com/pasxn/waffle/blob/main/extra/demo/inference.py) to understand the API.

Reading [unit tests](https://github.com/pasxn/waffle/tree/main/test) is another way to understand the whole workflow of the framework and the API.

## Adding an Operation

To add a custom operation to the framework, it should be implemented [here](https://github.com/pasxn/waffle/tree/main/waffle/nn) using the primitive operations in [ops](https://github.com/pasxn/waffle/blob/main/waffle/ops.py).

An operation is implemented as a class and should be then manually imported [here](https://github.com/pasxn/waffle/blob/main/waffle/nn/__init__.py).

Finally, support to of the custom operation should be added [here](https://github.com/pasxn/waffle/blob/main/waffle/onnx/node.py/#L18) to the ONNX node class.

## Adding an Accelerator

To add a custom accelerator to waffle, the operations specified [here](#primitive-operations) should be manually implemented as a hardware specific software component in a directory in [backend](https://github.com/pasxn/waffle/tree/main/waffle/backend). A Python API should also be implemented for the particular component in the backend module.

Then the operations can be manually mapped to each accelerator statically or dynamically from the functions specific to the primitive operations in [ops](https://github.com/pasxn/waffle/tree/main/waffle/ops.py).

Waffle currently only has a CPU backend. The backend and the API can be found [here](https://github.com/pasxn/waffle/tree/main/waffle/backend) for reference.