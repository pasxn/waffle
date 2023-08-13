<div align="center">

<img src="https://raw.githubusercontent.com/pasxn/waffle/docs/docs/assets/logo.jpg" alt="logo" width="180"/>

## waffle: a simple machine learning inference framework for the edge!

[Homepage](https://github.com/pasxn/waffle) | [Documentation](https://github.com/pasxn/waffle/blob/main/docs/abstractions.md) | [Examples](/examples)

</div>

This is an effort to learn how neural networks work by building a simple toy inference framework. Originally inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd). Micrograd works only on scalars. Waffle works on tensors and uses [Numpy](https://numpy.org/) under the hood. The original plan was to provide GPU acceleration by targeting the [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) SoC. But due to technical complexities the hardware acceleration code was made stale and moved [here](https://github.com/pasxn/v3dBLAS).

You sort of still can add accelerators to the architecture. More details can be found [here](https://github.com/pasxn/waffle/blob/main/docs/abstractions.md/#adding-an-accelerator).

### Features
#### Neural Networks

Waffle can run neural networks given only data and the ONNX graph assuming we support all the operations in your graph. More details 

### Installation

```bash
git clone https://github.com/pasxn/waffle.git
cd waffle
```
```bash
python -m pip install -e .
```
or if you wanna run tests,
```bash
python3 -m pip install -e '.[testing]'
```

### Running tests

```bash
python -m pytest -s -v
```