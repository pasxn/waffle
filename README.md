<div align="center">

<img src="https://raw.githubusercontent.com/pasxn/waffle/main/docs/assets/logo.png" alt="logo" width="160"/>

## waffle: a simple machine learning inference framework for the edge!

[Homepage](https://github.com/pasxn/waffle) | [Documentation](https://github.com/pasxn/waffle/blob/main/docs/documentation.md) | [Examples](/examples)

</div>

This is an effort to learn how neural networks work by writing a simple inference framework. This was originally inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd). Micrograd works only on scalars. Waffle works on tensors and for that, uses [Numpy](https://numpy.org) under the hood. The initial plan was to provide GPU acceleration by targeting the [BCM2711](https://datasheets.raspberrypi.com/bcm2711/bcm2711-peripherals.pdf) SoC in [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b). But due to technical complexities, hardware acceleration code was made stale and moved [here](https://github.com/pasxn/v3dBLAS).

You sort of still can add accelerators to the architecture. More details about that can be found [here](https://github.com/pasxn/waffle/blob/main/docs/documentation.md/#adding-an-accelerator).

### Features
#### Neural Networks

Waffle can run neural networks given only data and the ONNX graph assuming we support all the operations in your graph. You can even add your custom operations. More details about that can be found [here](https://github.com/pasxn/waffle/blob/main/docs/documentation.md/#adding-an-operation).

```py
from waffle import nn

model = nn.Module('your_model_name', './path/to/your/model.onnx')
model.compile()

# assuming you have data in the required format in an object called data
output = model.run(data)
```
#### Tensor Computation

We have written a tensor library with all the features required for us to run basic neural networks. Have a look at [tensor.py](https://github.com/pasxn/waffle/blob/main/waffle/tensor.py) for the exact details.

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
python -m pip install -e '.[testing]'
```

### Running Tests

```bash
python -m pytest -s -v
```

### License

[MIT](https://github.com/pasxn/waffle/blob/main/LICENSE)
