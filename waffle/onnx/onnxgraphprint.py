#%%
import onnx
import numpy as np

#%%
##grap prnt
model_path ='../../models/resnet18/resnet18.onnx'
model =onnx.load(model_path)
print(model)



#%%print ops inps and outps
graph = model.graph
for node in graph.node:
    print("Ops:", node.op_type)
    print("Inps:", node.input)
    print("Outs:", node.output)



# %%
#Weight extractn
# for initializer in model.graph.initializer
    #WeightName --- initializer.name 
    #WeightValues --- initializer.raw_data

#rawdata ------> NumPyArray
    #numpy.frombuffer(buffer=WeightValues, dtype=initializer.data_type)
    #                                     (dtype reshape Tensor to Numpy Aray)
    #                                     np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype(initializer.data_type)])

#Print 

for initializer in model.graph.initializer:
    WeightName = initializer.name
    WeightValues = initializer.raw_data

    WeightArray= np.frombuffer(buffer=WeightValues, dtype=np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]))
    
    print ('Name = ', WeightName)
    #print ('Weight Array = ', WeightArray)


#reshape thhe Weighht Array into thhe shhape of Weighht Tensor
    TensorArray=np.reshape(WeightArray, tuple(initializer.dims))   
    print('Tensor Array Shape: ', TensorArray.shape)
    
    assert TensorArray.shape == tuple(initializer.dims) 





# %% 
# A clear test to convert the onnx raw weigts to an weigt array resaping it to weigt shape
import onnx

# Load the ONNX model
model_path = '../../models/resnet18/resnet18.onnx'
model = onnx.load(model_path)

# Access the initializer list
initializer_list = model.graph.initializer

for initializer in initializer_list:
    # Access the name, shape, and values of the weight tensor
    weight_name = initializer.name
    weight_shape = initializer.dims
    weight_values = initializer.raw_data

    #raw_data to a NumPy array
    import numpy as np
    weight_array = np.frombuffer(weight_values, dtype=np.float32).reshape(weight_shape)

    # Print the weight information
    print("Weight Name:", weight_name)
    print("Weight Shape:", weight_shape)
    print("Weight Values:", weight_array)
