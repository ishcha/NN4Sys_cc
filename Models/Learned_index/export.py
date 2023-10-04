import numpy as np
import torch.onnx
import random
import os
import onnx
from models import LINN, DeepLINN


MODEL_LIST = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'parallel', 'concat']
#NN_MODEL = f'./gym/results/pcc_model_{i}_10_best.pt'
ONNX_DIR = f'../../Benchmarks/onnx'




def main():
    os.system(f'cp lindex.onnx {ONNX_DIR}/lindex.onnx')
    os.system(f'cp lindex_deep.onnx {ONNX_DIR}/lindex_deep.onnx')
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)


    actor = LINN(1,1)
    input = torch.zeros(1,1)
    save_path =  f'{ONNX_DIR}/lindex_tmp.onnx'
    torch_out = actor(input)
    print(torch_out)
    torch.onnx.export(actor,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      output_names=['output'])  # the model's output names
    actor = onnx.load(save_path)
    onnx.checker.check_model(actor)

    actor = DeepLINN(1,1)
    input = torch.zeros(1,1)
    save_path =  f'{ONNX_DIR}/lindex_deep_tmp.onnx'
    torch_out = actor(input)
    print(torch_out)
    torch.onnx.export(actor,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      output_names=['output'])  # the model's output names
    actor = onnx.load(save_path)
    onnx.checker.check_model(actor)



if __name__ == '__main__':
    main()
