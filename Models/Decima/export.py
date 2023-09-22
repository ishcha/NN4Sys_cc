import numpy as np
import torch.onnx
import random
import os
import onnx
import model_benchmark as model
from spark_env.env import Environment
from msg_passing_path import *
import bisect
from spark_env.job_dag import JobDAG
from spark_env.node import Node

ONNX_DIR = './benchmark/decima/onnxs'
MODEL_LIST = ['mid']
MODEL = MODEL_LIST[1]
MODEL_TYPES = ['simple', 'marabou']
MODEL_TYPE = MODEL_TYPES[2]
file_path = "./best_models/model_exec50_ep_" + str(6200)



def load_model(actor):
    actor.load_state_dict(torch.load(file_path + "gcn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "gsn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "actor.pth", map_location='cpu'), strict=False)
    actor.eval()
    return actor




def main():
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)
    save_path = ONNX_DIR + '/decima_' + MODEL + '_' + MODEL_TYPE + ".onnx"
    input_arrays = np.load(f'./benchmark/decima/decima_resources/decima_fixiedInput_1.npy')
    print(save_path)
    if MODEL_TYPE == 'simple':
        if MODEL == 'mid':

            actor = model.model_benchmark()
    if MODEL_TYPE == 'marabou':
        input = torch.tensor(input_arrays[0][:-1])

        actor = model.model_benchmark_marabou(input)



    actor = load_model(actor)
    actor = actor.eval()


    # get input
    node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = generate_input()

    number = len(gcn_mats)
    for i in range(number):
        gcn_mats[i] = gcn_mats[i].to_dense()


    if MODEL_TYPE == 'simple':
        input = torch.zeros(1, 4300).to(torch.float32)
    if MODEL_TYPE == 'concat':
        input = torch.zeros(1, 4600).to(torch.float32)



    torch_out = actor(cocnat_input)
    print("output:")
    print(torch_out)


    torch.onnx.export(actor,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      output_names=['output'])  # the model's output names

    # check the model
    actor = onnx.load(save_path)
    onnx.checker.check_model(actor)


if __name__ == '__main__':
    main()
