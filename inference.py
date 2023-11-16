import os
import numpy as np
import onnx
import onnxruntime as ort


# The directory of your input and output data
input_data_dir = './input_data'
output_data_dir = './output_data'
model_6 = onnx.load('fengwu.onnx')

# Set the behavier of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 1

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
ort_session_6 = ort.InferenceSession('fengwu.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])


data_mean = np.load("data_mean.npy")[:, np.newaxis, np.newaxis]
data_std = np.load("data_std.npy")[:, np.newaxis, np.newaxis]

input1 = np.load(os.path.join(input_data_dir, 'input1.npy')).astype(np.float32)
input2 = np.load(os.path.join(input_data_dir, 'input2.npy')).astype(np.float32)
# input1 = np.random.rand(69, 721, 1440)
# input2 = np.random.rand(69, 721, 1440)

input1_after_norm = (input1 - data_mean) / data_std
input2_after_norm = (input2 - data_mean) / data_std
input = np.concatenate((input1_after_norm, input2_after_norm), axis=0)[np.newaxis, :, :, :]
input = input.astype(np.float32)

for i in range(56):
    output = ort_session_6.run(None, {'input':input})[0]
    input = np.concatenate((input[:, 69:], output[:, :69]), axis=1)
    output = (output[0, :69] * data_std) + data_mean
    print(output.shape)
  # np.save(os.path.join(output_data_dir, f"output_{i}.npy"), output)       #保存输出