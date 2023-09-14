import sys

sys.path.append("../Models/Decima/")
import numpy as np
import torch.onnx
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
MODEL = MODEL_LIST[0]
MODEL_TYPES = ['simple', 'marabou']
MODEL_TYPE = MODEL_TYPES[0]
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
    print(save_path)
    if MODEL_TYPE == 'simple':
        if MODEL == 'mid':
            actor = model.model_benchmark()
        input = torch.zeros(1, 4300).to(torch.float32)
    if MODEL_TYPE == 'marabou':
        if MODEL == 'mid':
            actor = model.model_benchmark_marabou()

    print("load model")
    actor = load_model(actor)
    actor = actor.eval()

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
