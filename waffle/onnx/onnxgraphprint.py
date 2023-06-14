#%%
import onnx
import numpy as np

#%%
## for grap prnt
model_path ='../../models/resnet18/resnet18.onnx'
model =onnx.load(model_path)
print(model)



#%% for print ops inps and outps
graph = model.graph
for node in graph.node:
    print("Ops:", node.op_type)
    print("Inps:", node.input)
    print("Outs:", node.output)



# %% initial approach for weight extraction

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

#%%# %%----------------------------------------------------------------------------
#for input and Output Details

# Print input details
print("Input details:")
for i, input_node in enumerate(model.graph.input):
    print(f"Input Node {i}:")
    print("Name:", input_node.name)
    print("Shape:", input_node.type.tensor_type.shape.dim)
    print()

# Print output details
print("Output details:")
for i, output_node in enumerate(model.graph.output):
    print(f"Output Node {i}:")
    print("Name:", output_node.name)
    print("Shape:", output_node.type.tensor_type.shape.dim)
    print()
#-------------------------------------------------------------------------------


#%%just checking wethr B & W means Bias and Weights
import onnx

model = onnx.load('../../models/resnet18/resnet18.onnx')

# weight tensor (W)
for initializer in model.graph.initializer:
        if initializer.name == 'onnx::Conv_196':
            weight_tensor = initializer
            break

#  bias tensor (B)
for initializer in model.graph.initializer: 
        if initializer.name == 'onnx::Conv_197':
            bias_tensor = initializer
            break

print("Weight Tensor Shape:", weight_tensor.dims)
print("Bias Tensor Shape:", bias_tensor.dims)






#%% For Node Details including bias & Weights
import onnx
import numpy as np

# Load the ONNX model
model_path = '../../models/resnet18/resnet18.onnx'
model = onnx.load(model_path)

initializer_list = model.graph.initializer

# Print node details
print("Node Details:")
for i, node in enumerate(model.graph.node):
    print(f"========================================== Node {i}:==========================================")
    print("Name:", node.name)
    print()
    print("Input Nodes:", node.input)
    print()
    print("Output Nodes:", node.output)
    print()
    print("Op Type:", node.op_type)
    print()
    print("Attribute:")
    for attr in node.attribute:
        print(f"- Name: {attr.name}")
        print(f"  Type: {attr.type}")
        #attributes can be numerics (kernal size,shape,etc.)
        if attr.type == onnx.AttributeProto.FLOATS:
            print(f"  Values1: {attr.floats}")
        elif attr.type == onnx.AttributeProto.INTS:
            print(f"  Values2: {attr.ints}")
        #attributes can be stored in texts (Names)
        elif attr.type == onnx.AttributeProto.STRING:
            print(f"  Values3: {attr.s}")
        elif attr.type == onnx.AttributeProto.TENSOR:
            print(f" Tensor Value: ", attr.t.float_data)
        print()
        print()
    
    #Print Weights Also
    #for initializer in initializer_list:
        # Access the name, shape, and values of the weight tensor
        weight_name = initializer.name
        weight_shape = initializer.dims
        weight_values = initializer.raw_data

        #raw_data to a NumPy array
        import numpy as np
        weight_array = np.frombuffer(weight_values, dtype=np.float32).reshape(weight_shape)

        # Print the weight information
        print("Weight Name:", weight_name)
        print()
        print("Weight Shape:", weight_shape)
        print()
        #print("Weight Values:", weight_array)

print()
print()

# %%
