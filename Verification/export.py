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
MODEL_LIST = ['small', 'mid', 'big']
MODEL = MODEL_LIST[1]
MODEL_TYPES = ['simple', 'parallel', 'concat']
MODEL_TYPE = MODEL_TYPES[2]
file_path = "./best_models/model_exec50_ep_" + str(6200)


def translate_state(obs):
    """
    Translate the observation to matrix form
    """
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map = obs

    # compute total number of nodes
    total_num_nodes = int(np.sum(job_dag.num_nodes for job_dag in job_dags))

    # job and node inputs to feed
    node_inputs = np.zeros([total_num_nodes, 5])
    job_inputs = np.zeros([len(job_dags), 3])

    # sort out the exec_map
    exec_map = {}
    for job_dag in job_dags:
        exec_map[job_dag] = len(job_dag.executors)
    # count in moving executors
    for node in moving_executors.moving_executors.values():
        exec_map[node.job_dag] += 1
    # count in executor commit
    for s in exec_commit.commit:
        if isinstance(s, JobDAG):
            j = s
        elif isinstance(s, Node):
            j = s.job_dag
        elif s is None:
            j = None
        else:
            print('source', s, 'unknown')
            exit(1)
        for n in exec_commit.commit[s]:
            if n is not None and n.job_dag != j:
                exec_map[n.job_dag] += exec_commit.commit[s][n]

    # gather job level inputs
    job_idx = 0
    for job_dag in job_dags:
        # number of executors in the job
        job_inputs[job_idx, 0] = exec_map[job_dag] / 20.0
        # the current executor belongs to this job or not
        if job_dag is source_job:
            job_inputs[job_idx, 1] = 2
        else:
            job_inputs[job_idx, 1] = -2
        # number of source executors
        job_inputs[job_idx, 2] = num_source_exec / 20.0

        job_idx += 1

    # gather node level inputs
    node_idx = 0
    job_idx = 0
    for job_dag in job_dags:
        for node in job_dag.nodes:
            # copy the feature from job_input first
            node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

            # work on the node
            node_inputs[node_idx, 3] = \
                (node.num_tasks - node.next_task_idx) * \
                node.tasks[-1].duration / 100000.0

            # number of tasks left
            node_inputs[node_idx, 4] = \
                (node.num_tasks - node.next_task_idx) / 200.0

            node_idx += 1

        job_idx += 1

    return node_inputs, job_inputs, \
           job_dags, source_job, num_source_exec, \
           frontier_nodes, executor_limits, \
           exec_commit, moving_executors, \
           exec_map, action_map


def get_valid_masks(job_dags, frontier_nodes,
                    source_job, num_source_exec, exec_map, action_map):
    executor_levels = range(1, 51)
    job_valid_mask = np.zeros([1,
                               len(job_dags) * len(range(1, 51))])

    job_valid = {}  # if job is saturated, don't assign node

    base = 0
    for job_dag in job_dags:
        # new executor level depends on the source of executor
        if job_dag is source_job:
            least_exec_amount = \
                exec_map[job_dag] - num_source_exec + 1
            # +1 because we want at least one executor
            # for this job
        else:
            least_exec_amount = exec_map[job_dag] + 1
            # +1 because of the same reason above

        assert least_exec_amount > 0
        assert least_exec_amount <= executor_levels[-1] + 1

        # find the index for first valid executor limit
        exec_level_idx = bisect.bisect_left(
            executor_levels, least_exec_amount)

        if exec_level_idx >= len(executor_levels):
            job_valid[job_dag] = False
        else:
            job_valid[job_dag] = True

        for l in range(exec_level_idx, len(executor_levels)):
            job_valid_mask[0, base + l] = 1

        base += executor_levels[-1]

    total_num_nodes = int(np.sum(
        job_dag.num_nodes for job_dag in job_dags))

    node_valid_mask = np.zeros([1, total_num_nodes])

    for node in frontier_nodes:
        if job_valid[node.job_dag]:
            act = action_map.inverse_map[node]
            node_valid_mask[0, act] = 1

    return node_valid_mask, job_valid_mask


def generate_input():
    env = Environment()

    # reset environment
    env.seed(0)
    env.reset()

    obs = env.observe()

    node_inputs, job_inputs, \
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, \
    exec_map, action_map = translate_state(obs)

    postman = Postman()

    # get message passing path (with cache)
    gcn_mats, gcn_masks, dag_summ_backward_map, \
    running_dags_mat, job_dags_changed = \
        postman.get_msg_path(job_dags)

    # get node and job valid masks
    node_valid_mask, job_valid_mask = \
        get_valid_masks(job_dags, frontier_nodes,
                        source_job, num_source_exec, exec_map, action_map)

    # get summarization path that ignores finished nodes
    summ_mats = get_unfinished_nodes_summ_mat(job_dags)

    node_inputs = torch.Tensor(node_inputs).to(torch.float32)
    job_inputs = torch.tensor(job_inputs).to(torch.float32)
    node_valid_mask = torch.tensor(node_valid_mask).to(torch.float32)
    job_valid_mask = torch.tensor(job_valid_mask).to(torch.float32)
    gcn_masks = torch.tensor(np.array(gcn_masks)).to(torch.float32)
    dag_summ_backward_map = torch.Tensor(dag_summ_backward_map).to(torch.float32)

    summ_mats = summ_mats.to(torch.float32)
    running_dags_mat = running_dags_mat.to(torch.float32)

    return node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map


def load_model(actor):
    actor.load_state_dict(torch.load(file_path + "gcn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "gsn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "actor.pth", map_location='cpu'), strict=False)
    actor.eval()
    return actor


def write_vnnlib(node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map):
    if not os.path.exists("./benchmark/decima/vnnlib"):
        os.makedirs("./benchmark/decima/vnnlib")
    with open("./benchmark/decima/vnnlib/decima_test.vnnlib", "w") as f:
        f.write("\n")

        print("=====")

        print(node_inputs.size())
        print(node_valid_mask.size())
        print(gcn_mats.size())
        print(gcn_masks.size())
        print(summ_mats.size())
        print(running_dags_mat.size())
        print(dag_summ_backward_map.size())

        node_inputs = torch.flatten(node_inputs)
        node_valid_mask = node_valid_mask.flatten()
        gcn_mats = gcn_mats.flatten()
        gcn_masks = gcn_masks.flatten()
        summ_mats = summ_mats.flatten()
        running_dags_mat = running_dags_mat.flatten()
        dag_summ_backward_map = dag_summ_backward_map.flatten()

        index = 0
        number = node_inputs.size()[0]
        print("=====")
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = node_valid_mask.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = gcn_mats.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = gcn_masks.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = summ_mats.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = running_dags_mat.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1

        number = dag_summ_backward_map.size()[0]
        print(number)
        for i in range(number):
            f.write(f"(declare-const X_{index} Real)\n")
            index += 1
        for i in range(7):
            f.write(f"(declare-const Y_{i} Real)\n")

        index = 0
        number = node_inputs.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = node_valid_mask.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = gcn_mats.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = gcn_masks.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = summ_mats.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = running_dags_mat.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1

        number = dag_summ_backward_map.size()[0]
        for i in range(number):
            f.write(f"(assert (>= X_{index} 0))\n")
            f.write(f"(assert (<= X_{index} 0))\n")
            index += 1
        for i in range(7):
            f.write(f"(assert (<= Y_{i} 0))\n")
            f.write(f"(assert (>= Y_{i} 0))\n")


def write_vnnlib_temp():
    if not os.path.exists("./benchmark/decima/vnnlib"):
        os.makedirs("./benchmark/decima/vnnlib")
    with open("./benchmark/decima/vnnlib/decima_test.vnnlib", "w") as f:
        f.write("\n")

        for i in range(7681):
            f.write(f"(declare-const X_{i} Real)\n")

        for i in range(30):
            f.write(f"(declare-const Y_{i} Real)\n")

        for i in range(7681):
            f.write(f"(assert (>= X_{i} 0))\n")
            f.write(f"(assert (<= X_{i} 0))\n")

        for i in range(30):
            f.write(f"(assert (<= Y_{i} 0))\n")
            f.write(f"(assert (>= Y_{i} 0))\n")


def main():
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)
    save_path = ONNX_DIR + '/decima_' + MODEL + '_' + MODEL_TYPE + ".onnx"
    input_arrays = np.load(f'./benchmark/decima/decima_resources/decima_fixiedInput_1.npy')
    print(save_path)
    if MODEL_TYPE == 'simple':
        if MODEL == 'mid':
            input = torch.tensor(input_arrays[0][:-1])
            print(input)
            actor = model.model_benchmark_marabou(input)
        if MODEL == 'big':
            actor = model.model_benchmark()
        if MODEL == 'small':
            actor = model.model_benchmark()
    print("load")
    if MODEL_TYPE == 'concat':
        if MODEL == 'mid':
            input_arrays = np.load(f'./benchmark/decima/decima_resources/decima_fixiedInput_3.npy')
            input_array = input_arrays[0]
            print(len(input_array))
            test_input = np.concatenate([input_array[:4300], input_array[4321:]])
            print(test_input)
            cocnat_input = torch.tensor(test_input)
            print(cocnat_input.size())
            actor = model.model_benchmark_concat_marabou(cocnat_input)
        if MODEL == 'big':
            actor = model.model_benchmark()
        if MODEL == 'small':
            actor = model.model_benchmark()
        # load model
    actor = load_model(actor)
    actor = actor.eval()
    print("load")

    # get input
    node_inputs, job_inputs, node_valid_mask, job_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = generate_input()

    number = len(gcn_mats)
    for i in range(number):
        gcn_mats[i] = gcn_mats[i].to_dense()
    gcn_mats = torch.stack(gcn_mats)
    summ_mats = summ_mats.to_dense()
    running_dags_mat = running_dags_mat.to_dense()

    #write_vnnlib(node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat,
    #             dag_summ_backward_map)
    if MODEL_TYPE == 'simple':
        input = torch.zeros(1, 4300).to(torch.float32)
    if MODEL_TYPE == 'concat':
        input = torch.zeros(1, 4600).to(torch.float32)

    write_vnnlib_temp()

    # run one time to test torch_out = actor(node_inputs, node_valid_mask,gcn_mats, gcn_masks, summ_mats,
    # running_dags_mat, dag_summ_backward_map)
    input = torch.tensor(input[0][:-1])
    torch_out = actor(cocnat_input)
    print("output:")
    print(torch_out)

    torch.onnx.export(actor,  # model being run
                      cocnat_input,  # model input (or a tuple for multiple inputs)
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
