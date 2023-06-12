#%%
import onnx
import numpy as np

# Load the ONNX file
model = onnx.load('mnist_cnn.onnx')

# Iterate over the graph's initializers to extract weights
for initializer in model.graph.initializer:
    # Extract the name and values of the weight tensor
    weight_name = initializer.name
    weight_values = initializer.raw_data
    

    # Decode the raw data into a numpy array
    dtype = initializer.data_type
        #To map the ONNX data type of the weight tensor to the corresponding NumPy data type
    weight_array = np.frombuffer(weight_values, dtype=np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]))
    
    # Print the weight name and values
    print("Weight Name:", weight_name)
    print("Weight Values:", weight_array.shape)
    
#   Weight Shape needed to extract
#   Do we need to reshape the weight array???

# %%
