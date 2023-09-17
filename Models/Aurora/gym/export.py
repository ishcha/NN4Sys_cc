import numpy as np
import torch.onnx
import random
import os
import onnx
import model_benchmark as model

ONNX_DIR = '../benchmark/aurora/onnxs'
MODEL_LIST = ['small', 'mid', 'big']
MODEL = MODEL_LIST[1]
MODEL_TYPES = ['simple', 'parallel', 'concat']
MODEL_TYPE = MODEL_TYPES[2]
NN_MODEL = './results/pcc_model_mid_10_4.pt'


def load_model(actor):
    para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
    newpara = {}
    newpara['policy_net.0.weight'] = para["mlp_extractor.policy_net.0.weight"]
    newpara['policy_net.0.bias'] = para["mlp_extractor.policy_net.0.bias"]
    newpara['policy_net.1.weight'] = para["mlp_extractor.policy_net.1.weight"]
    newpara['policy_net.1.bias'] = para["mlp_extractor.policy_net.1.bias"]
    newpara['policy_net.2.weight'] = para["mlp_extractor.policy_net.2.weight"]
    newpara['policy_net.2.bias'] = para["mlp_extractor.policy_net.2.bias"]
    actor.load_state_dict(newpara)
    actor.eval()
    return actor


def main():
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)
    save_path = ONNX_DIR + '/aurora_' + MODEL + '_' + MODEL_TYPE + ".onnx"
    print(save_path)
    if MODEL_TYPE == 'simple':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid()
        if MODEL == 'big':
            actor = model.CustomNetwork_big()
        if MODEL == 'small':
            actor = model.CustomNetwork_small()

        # load model
        actor = load_model(actor)

        # export
        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
        actor.load_state_dict(para, strict=False)
        actor = actor.eval()

        # run one time to test
        myinput = torch.zeros(1, 30).to(torch.float32)
        torch_out = actor(myinput)
        print(torch_out)

        torch.onnx.export(actor,  # model being run
                          myinput,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          output_names=['output'])  # the model's output names

    if MODEL_TYPE == 'parallel':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid_parallel()
        if MODEL == 'big':
            actor = model.CustomNetwork_big_parallel()
        if MODEL == 'small':
            actor = model.CustomNetwork_small_parallel()

        # load model
        actor = load_model(actor)

        # export
        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
        actor.load_state_dict(para, strict=False)
        actor = actor.eval()

        # run one time to test
        input_arrays = np.load(f'./benchmark/aurora/aurora_resources/aurora_fixiedInput_4.npy')
        print(input_arrays)
        myinput = input_arrays[0]
        print(myinput)
        myinput = torch.tensor(myinput)
        torch_out = actor(myinput)
        print(torch_out)

        torch.onnx.export(actor,  # model being run
                          myinput,  # model input (or a tuple for multiple inputs)
                          save_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          output_names=['output'])  # the model's output names

    if MODEL_TYPE == 'concat':
        if MODEL == 'mid':
            actor = model.CustomNetwork_mid_concatnate()
        if MODEL == 'big':
            actor = model.CustomNetwork_big_concatnate()
        if MODEL == 'small':
            actor = model.CustomNetwork_small_concatnate()

        # load model
        actor = load_model(actor)

        # export
        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
        actor.load_state_dict(para, strict=False)
        actor = actor.eval()

        # run one time to test


        # run one time to test
        myinput = torch.zeros(1, 151).to(torch.float32)
        myinput[0][150] = 1
        torch_out = actor(myinput)
        print(torch_out)

        torch.onnx.export(actor,  # model being run
                          myinput,  # model input (or a tuple for multiple inputs)
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
