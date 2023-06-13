#%%
import onnx
import numpy as np

#%%
##grap prnt
model_path ='../../models/resnet18/resnet18.onnx'
model =onnx.load(model_path)
print(model)



#%%ops inps and outps
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
    #WeightArray= np.frombuffer(buffer=WeightValues, dtype=[initializer.data_type])
    print ('Name = ', WeightName)
    #print ('Weight Array = ', WeightArray.shape)

    TensorArray=np.reshape(WeightArray, tuple(initializer.dims))
    print('Tensor Array Shape: ', TensorArray.shape)
    
    assert TensorArray.shape == tuple(initializer.dims)  #*********#












# %%

#for initializer in model.graph.initializer:
for initializer in model.graph.initializer:
    print ('Weight Array shape = ', WeightArray.shape)
    TensorArray=np.reshape(WeightArray, tuple(initializer.dims))
    print('Tensor Array Shape: ', TensorArray.shape)



# %%
# 
for initializer in model.graph.initializer:
    tensor = initializer
    print("Weight tensor shape:", tensor.dims)
    print('........')


#if len(tensor.float_data) > 0 else tensor.raw_data
# %%
for initializer in model.graph.initializer:
    print (initializer.dims)
