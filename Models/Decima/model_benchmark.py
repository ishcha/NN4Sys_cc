import torch.nn
import torch.nn as nn

from graph_convolution import GraphLayer
from torch.nn.parameter import Parameter

Max_Node = 20


class model_benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn
        self.output_dim = 8
        self.max_depth = 8

        self.act_fn = torch.nn.LeakyReLU()

        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network
        self.node_input_dim = 5
        self.job_input_dim = 3
        self.output_dim = 8
        self.executor_levels = range(1, 16)

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        node_inputs = input[:, :100].view([-1, Max_Node, 5])
        node_valid_mask = input[:, 100:120].view([-1, 1, Max_Node])
        gcn_mats = input[:, 120:3320].view([-1, 8, Max_Node, Max_Node])
        gcn_masks = input[:, 3320:3480].view([-1, 8, Max_Node, 1])
        summ_mats = input[:, 3480:3880].view([-1, Max_Node, Max_Node])
        running_dags_mat = input[:, 3880:3900].view([-1, 1, Max_Node])
        dag_summ_backward_map = input[:, 3900:4300].view([-1, Max_Node, Max_Node])

        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        y = y * gcn_masks[:, 0]

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 1]

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 2]

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 3]

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 4]

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 5]

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 6]

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 7]

        # assemble neighboring information
        x = x + y
        gcn_output = x

        # gsn
        x = torch.concat([node_inputs, gcn_output], dim=2)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.concat([
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary], dim=1)
        merge_node = torch.concat([
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)
        node_outputs = node_outputs.reshape(-1, 20)

        return node_outputs

class model_benchmark_split(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn
        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn
        self.act_fn = torch.nn.ReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=1)
        node_inputs = node_inputs.reshape([Max_Node, 5])
        node_valid_mask = node_valid_mask.reshape([-1, 1, Max_Node])
        gcn_mats = gcn_mats.reshape([8, Max_Node, Max_Node])
        gcn_masks = gcn_masks.reshape([8, Max_Node, 1])
        summ_mats = summ_mats.reshape([-1, Max_Node, Max_Node])
        running_dags_mat = running_dags_mat.reshape([-1, 1, Max_Node])
        dag_summ_backward_map = dag_summ_backward_map.reshape([-1, Max_Node, Max_Node])

        gcn_mats0, gcn_mats1, gcn_mats2, gcn_mats3, gcn_mats4, gcn_mats5, gcn_mats6, gcn_mats7 = torch.split(gcn_mats,
                                                                                                             [1, 1, 1,
                                                                                                              1, 1, 1,
                                                                                                              1, 1],
                                                                                                             dim=0)
        gcn_mats0 = gcn_mats0.reshape([20, 20])
        gcn_mats1 = gcn_mats1.reshape([20, 20])
        gcn_mats2 = gcn_mats2.reshape([20, 20])
        gcn_mats3 = gcn_mats3.reshape([20, 20])
        gcn_mats4 = gcn_mats4.reshape([20, 20])
        gcn_mats5 = gcn_mats5.reshape([20, 20])
        gcn_mats6 = gcn_mats6.reshape([20, 20])
        gcn_mats7 = gcn_mats7.reshape([20, 20])

        gcn_masks0, gcn_masks1, gcn_masks2, gcn_masks3, gcn_masks4, gcn_masks5, gcn_masks6, gcn_masks7 = torch.split(
            gcn_masks,
            [1, 1, 1,
             1, 1, 1,
             1, 1],
            dim=0)
        gcn_masks0 = gcn_masks0.reshape([20, 1])
        gcn_masks1 = gcn_masks1.reshape([20, 1])
        gcn_masks2 = gcn_masks2.reshape([20, 1])
        gcn_masks3 = gcn_masks3.reshape([20, 1])
        gcn_masks4 = gcn_masks4.reshape([20, 1])
        gcn_masks5 = gcn_masks5.reshape([20, 1])
        gcn_masks6 = gcn_masks6.reshape([20, 1])
        gcn_masks7 = gcn_masks7.reshape([20, 1])

        gcn_masks0 = torch.zeros(20, 1)
        gcn_masks1 = torch.zeros(20, 1)
        gcn_masks2 = torch.zeros(20, 1)
        gcn_masks3 = torch.zeros(20, 1)
        gcn_masks4 = torch.zeros(20, 1)
        gcn_masks5 = torch.zeros(20, 1)
        gcn_masks6 = torch.zeros(20, 1)
        gcn_masks7 = torch.zeros(20, 1)

        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats0, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks0

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats1, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks1

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats2, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks2

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats3, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks3

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats4, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks4

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats5, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks5

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats6, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks6

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        # y = torch.matmul(gcn_mats7, y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks7

        # assemble neighboring information
        x = x + y
        gcn_output = x

        # gsn
        print(node_inputs.size())
        print(gcn_output.size())
        x = torch.cat((node_inputs, gcn_output), dim=1)
        print("X", x.size())

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.cat((
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary), dim=1)

        node_inputs = node_inputs.view([1, 20, 5])
        gcn_output = gcn_output.view([1, 20, 8])
        merge_node = torch.cat((
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node), dim=2)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)
        node_outputs = torch.flatten(node_outputs, start_dim=1)

        return node_outputs

class model_benchmark_marabou(nn.Module):
    def __init__(self, input):
        super().__init__()
        # gcn
        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn
        self.act_fn = torch.nn.ReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.relu = nn.ReLU()
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=1)
        self.gcn_mats = gcn_mats.reshape([8, Max_Node, Max_Node])
        self.gcn_masks = gcn_masks.reshape([8, Max_Node, 1])
        gcn_mats0, gcn_mats1, gcn_mats2, gcn_mats3, gcn_mats4, gcn_mats5, gcn_mats6, gcn_mats7 = torch.split(gcn_mats,
                                                                                                             [1, 1, 1,
                                                                                                              1, 1, 1,
                                                                                                              1, 1],
                                                                                                             dim=0)
        gcn_masks0, gcn_masks1, gcn_masks2, gcn_masks3, gcn_masks4, gcn_masks5, gcn_masks6, gcn_masks7 = torch.split(
            gcn_masks,
            [1, 1, 1,
             1, 1, 1,
             1, 1],
            dim=0)
        self.summ_mats = summ_mats.reshape([-1, Max_Node, Max_Node])
        self.running_dags_mat = running_dags_mat.reshape([-1, 1, Max_Node])
        self.dag_summ_backward_map = dag_summ_backward_map.reshape([-1, Max_Node, Max_Node])
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=0)
        self.gcn_mats = Parameter(gcn_mats.reshape([8, Max_Node, Max_Node]))
        self.gcn_masks = Parameter(gcn_masks.reshape([8, Max_Node, 1]))
        self.summ_mats = Parameter(summ_mats.reshape([Max_Node, Max_Node]))
        self.running_dags_mat = Parameter(running_dags_mat.reshape([1, Max_Node]))
        self.dag_summ_backward_map = Parameter(dag_summ_backward_map.reshape([Max_Node, Max_Node]))

    def forward(self, input):
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=0)
        node_inputs = node_inputs.reshape([Max_Node, 5])

        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[0]

        # assemble neighboring information
        y = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[1]

        # assemble neighboring information
        y = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[2]

        # assemble neighboring information
        y = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[3]

        # assemble neighboring information
        y = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[4]

        # assemble neighboring information
        y = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[7]

        # assemble neighboring information
        y = x + y
        gcn_output = y

        # gsn

        x = torch.cat((node_inputs, gcn_output), dim=1)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(self.summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(self.running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(self.dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.cat((
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary), dim=0)

        # node_inputs = node_inputs.view([1, 20, 5])
        # gcn_output = gcn_output.view([1, 20, 8])

        merge_node = torch.cat((
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node), dim=1)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        # valid mask on node
        node_valid_mask = node_valid_mask * 10000.0
        node_outputs = node_outputs.reshape(1, 20)

        # apply mask
        node_outputs = node_outputs + node_valid_mask




        return node_outputs

class model_benchmark_concat_marabou(nn.Module):
    def __init__(self, input):
        super().__init__()
        # gcn
        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn
        self.act_fn = torch.nn.ReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network

        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=-1)
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map = torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400], dim=1)
        self.gcn_mats = gcn_mats.reshape([8, Max_Node, Max_Node])
        self.gcn_masks = gcn_masks.reshape([8, Max_Node, 1])
        gcn_mats0, gcn_mats1, gcn_mats2, gcn_mats3, gcn_mats4, gcn_mats5, gcn_mats6, gcn_mats7 = torch.split(gcn_mats,
                                                                                                             [1, 1, 1,
                                                                                                              1, 1, 1,
                                                                                                              1, 1],
                                                                                                             dim=0)
        gcn_masks0, gcn_masks1, gcn_masks2, gcn_masks3, gcn_masks4, gcn_masks5, gcn_masks6, gcn_masks7 = torch.split(
            gcn_masks,
            [1, 1, 1,
             1, 1, 1,
             1, 1],
            dim=0)
        self.summ_mats = summ_mats.reshape([-1, Max_Node, Max_Node])
        self.running_dags_mat = running_dags_mat.reshape([-1, 1, Max_Node])
        self.dag_summ_backward_map = dag_summ_backward_map.reshape([-1, Max_Node, Max_Node])
        '''
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map,\
            node_to_job, job_to_node= torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400,20,60], dim=0)
        self.gcn_mats = Parameter(gcn_mats.reshape([8, Max_Node, Max_Node]))
        self.gcn_masks = Parameter(gcn_masks.reshape([8, Max_Node, 1]))
        self.summ_mats = Parameter(summ_mats.reshape([Max_Node, Max_Node]))
        self.running_dags_mat = Parameter(running_dags_mat.reshape([1, Max_Node]))
        self.dag_summ_backward_map = Parameter(dag_summ_backward_map.reshape([Max_Node, Max_Node]))

    def forward(self, input):
        node_inputs, node_valid_mask, gcn_mats, gcn_masks, summ_mats, running_dags_mat, dag_summ_backward_map,\
            node_to_job, job_to_node= torch.split(
            input, [100, 20, 3200, 160, 400, 20, 400,20,60], dim=0)
        node_inputs = node_inputs.reshape([Max_Node, 5])

        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[0]

        # assemble neighboring information
        y = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[1]

        # assemble neighboring information
        y = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[2]

        # assemble neighboring information
        y = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[3]

        # assemble neighboring information
        y = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[4]

        # assemble neighboring information
        y = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[7]

        # assemble neighboring information
        y = x + y
        gcn_output = y

        # gsn

        x = torch.cat((node_inputs, gcn_output), dim=1)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(self.summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(self.running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(self.dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.cat((
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary), dim=0)


        merge_node = torch.cat((
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node), dim=1)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        # valid mask on node
        node_valid_mask = node_valid_mask * 10000.0
        node_outputs = node_outputs.reshape(1, 20)


        # apply mask
        node_outputs_1 = node_outputs + node_valid_mask



        #x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19=torch.split(node_outputs,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dim=1)
        #mysum = x1+x2+x3+x4

        #chose = x1*0
        #print(chose.dtype)


        #chosen_jobs = node_to_job[chose]

        #node_index1 = (chosen_jobs * 20 + 0).to(torch.int)
        #node1 = job_to_node[node_index1].to(torch.int)
        node_inputs[0, 0] += 1 / 15
        node_inputs[0, 2] -= 1 / 15

        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[0]

        # assemble neighboring information
        y = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[1]

        # assemble neighboring information
        y = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[2]

        # assemble neighboring information
        y = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[3]

        # assemble neighboring information
        y = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[4]

        # assemble neighboring information
        y = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        # y = y * self.gcn_masks[5]

        # assemble neighboring information
        y = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        # y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(self.gcn_mats[7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        # y = y * self.gcn_masks[7]

        # assemble neighboring information
        y = x + y
        gcn_output = y

        # gsn

        x = torch.cat((node_inputs, gcn_output), dim=1)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(self.summ_mats, s)

        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)

        gsn_global_summary = torch.matmul(self.running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(self.dag_summ_backward_map, gsn_dag_summary)

        gsn_global_summ_extend_node = torch.cat((
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary), dim=0)

        merge_node = torch.cat((
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node), dim=1)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        # valid mask on node
        node_valid_mask = node_valid_mask * 10000.0
        node_outputs = node_outputs.reshape(1, 20)

        # apply mask
        node_outputs_2 = node_outputs + node_valid_mask

        return torch.concat([node_outputs_1,node_outputs_2,node_outputs_2])

class model_concat_benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        # gcn
        self.act_fn = torch.nn.LeakyReLU()

        # initialize message passing transformation parameters
        # h: x -> x'
        self.h_gc1 = GraphLayer(5, 16)
        self.h_gc2 = GraphLayer(16, 8)
        self.h_gc3 = GraphLayer(8, 8)

        # f: x' -> e
        self.f_gc1 = GraphLayer(8, 16)
        self.f_gc2 = GraphLayer(16, 8)
        self.f_gc3 = GraphLayer(8, 8)

        # g: e -> e
        self.g_gc1 = GraphLayer(8, 16)
        self.g_gc2 = GraphLayer(16, 8)
        self.g_gc3 = GraphLayer(8, 8)

        # gsn
        self.act_fn = torch.nn.LeakyReLU()

        # initialize summarization parameters for each hierarchy
        self.dag_gc1 = GraphLayer(5 + 8, 16)
        self.dag_gc2 = GraphLayer(16, 8)
        self.dag_gc3 = GraphLayer(8, 8)

        self.global_gc1 = GraphLayer(8, 16)
        self.global_gc2 = GraphLayer(16, 8)
        self.global_gc3 = GraphLayer(8, 8)

        # actor network
        self.fc1 = nn.Linear(29, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        node_inputs = input[:, :100].view([-1, Max_Node, 5])
        node_valid_mask = input[:, 100:120].view([-1, 1, Max_Node])
        gcn_mats = input[:, 120:3320].view([-1, 8, Max_Node, Max_Node])
        gcn_masks = input[:, 3320:3480].view([-1, 8, Max_Node, 1])
        summ_mats = input[:, 3480:3880].view([-1, Max_Node, Max_Node])
        running_dags_mat = input[:, 3880:3900].view([-1, 1, Max_Node])
        dag_summ_backward_map = input[:, 3900:4300].view([-1, Max_Node, Max_Node])

        node_to_job = input[:, 4300:4320]

        job_to_node = input[:, 4320:4380]

        # step 1
        # gcn
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        y = y * gcn_masks[:, 0]

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 1]

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 2]

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 3]

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 4]

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 5]

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 6]

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 7]

        # assemble neighboring information
        x = x + y
        gcn_output = x

        # gsn
        x = torch.concat([node_inputs, gcn_output], dim=2)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)
        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)
        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)
        gsn_global_summ_extend_node = torch.concat([
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary], dim=1).to(torch.float32)
        merge_node = torch.concat([
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2).to(torch.float32)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)
        node_outputs = torch.pow(node_outputs, 5)
        node_outputs = node_outputs / torch.sum(node_outputs)
        node_outputs = torch.pow(node_outputs, 5)
        node_outputs = node_outputs / torch.sum(node_outputs)
        node_outputs = node_outputs.reshape(-1, 20)
        node_outputs1 = node_outputs
        chose = torch.sum(torch.arange(20) * node_outputs, 1)
        chose = torch.round(chose).to(torch.int)

        # step 2
        # gcn
        # modify input
        print(chose)
        print(node_to_job)
        chosen_jobs = node_to_job[:, chose]

        node_index1 = (chosen_jobs * 20 + 0).to(torch.int)
        node1 = job_to_node[:, node_index1].to(torch.int)
        node_inputs[:, node1, 0] += 1 / 15
        node_inputs[:, node1, 2] -= 1 / 15

        node_index2 = (chosen_jobs * 20 + 1).to(torch.int)
        node2 = job_to_node[:, node_index2].to(torch.int)
        node_inputs[:, node2, 0] += 1 / 15
        node_inputs[:, node2, 2] -= 1 / 15

        node_index3 = (chosen_jobs * 20 + 2).to(torch.int)
        node3 = job_to_node[:, node_index3].to(torch.int)
        node_inputs[:, node3, 0] += 1 / 15
        node_inputs[:, node3, 2] -= 1 / 15

        node_index4 = (chosen_jobs * 20 + 3).to(torch.int)
        node4 = job_to_node[:, node_index4].to(torch.int)
        node_inputs[:, node4, 0] += 1 / 15
        node_inputs[:, node4, 2] -= 1 / 15

        node_index5 = (chosen_jobs * 20 + 4).to(torch.int)
        node5 = job_to_node[:, node_index5].to(torch.int)
        node_inputs[:, node5, 0] += 1 / 15
        node_inputs[:, node5, 2] -= 1 / 15

        node_index6 = (chosen_jobs * 20 + 5).to(torch.int)
        node6 = job_to_node[:, node_index6].to(torch.int)
        node_inputs[:, node6, 0] += 1 / 15
        node_inputs[:, node6, 2] -= 1 / 15

        node_index7 = (chosen_jobs * 20 + 6).to(torch.int)
        node7 = job_to_node[:, node_index7].to(torch.int)
        node_inputs[:, node7, 0] += 1 / 15
        node_inputs[:, node7, 2] -= 1 / 15

        node_index8 = (chosen_jobs * 20 + 7).to(torch.int)
        node8 = job_to_node[:, node_index8].to(torch.int)
        node_inputs[:, node8, 0] += 1 / 15
        node_inputs[:, node8, 2] -= 1 / 15

        node_index9 = (chosen_jobs * 20 + 8).to(torch.int)
        node9 = job_to_node[:, node_index9].to(torch.int)
        node_inputs[:, node9, 0] += 1 / 15
        node_inputs[:, node9, 2] -= 1 / 15

        node_index10 = (chosen_jobs * 20 + 9).to(torch.int)
        node10 = job_to_node[:, node_index10].to(torch.int)
        node_inputs[:, node10, 0] += 1 / 15
        node_inputs[:, node10, 2] -= 1 / 15

        node_index11 = (chosen_jobs * 20 + 10).to(torch.int)
        node11 = job_to_node[:, node_index11].to(torch.int)
        node_inputs[:, node11, 0] += 1 / 15
        node_inputs[:, node11, 2] -= 1 / 15

        node_index12 = (chosen_jobs * 20 + 11).to(torch.int)
        node12 = job_to_node[:, node_index12].to(torch.int)
        node_inputs[:, node12, 0] += 1 / 15
        node_inputs[:, node12, 2] -= 1 / 15

        node_index13 = (chosen_jobs * 20 + 12).to(torch.int)
        node13 = job_to_node[:, node_index13].to(torch.int)
        node_inputs[:, node13, 0] += 1 / 15
        node_inputs[:, node13, 2] -= 1 / 15

        node_index14 = (chosen_jobs * 20 + 13).to(torch.int)
        node14 = job_to_node[:, node_index14].to(torch.int)
        node_inputs[:, node14, 0] += 1 / 15
        node_inputs[:, node14, 2] -= 1 / 15

        node_index15 = (chosen_jobs * 20 + 14).to(torch.int)
        node15 = job_to_node[:, node_index15].to(torch.int)
        node_inputs[:, node15, 0] += 1 / 15
        node_inputs[:, node15, 2] -= 1 / 15

        node_index16 = (chosen_jobs * 20 + 15).to(torch.int)
        node16 = job_to_node[:, node_index16].to(torch.int)
        node_inputs[:, node16, 0] += 1 / 15
        node_inputs[:, node16, 2] -= 1 / 15

        node_index17 = (chosen_jobs * 20 + 16).to(torch.int)
        node17 = job_to_node[:, node_index17].to(torch.int)
        node_inputs[:, node17, 0] += 1 / 15
        node_inputs[:, node17, 2] -= 1 / 15

        node_index18 = (chosen_jobs * 20 + 17).to(torch.int)
        node18 = job_to_node[:, node_index18].to(torch.int)
        node_inputs[:, node18, 0] += 1 / 15
        node_inputs[:, node18, 2] -= 1 / 15

        node_index19 = (chosen_jobs * 20 + 18).to(torch.int)
        node19 = job_to_node[:, node_index19].to(torch.int)
        node_inputs[:, node19, 0] += 1 / 15
        node_inputs[:, node19, 2] -= 1 / 15

        node_index20 = (chosen_jobs * 20 + 19).to(torch.int)
        node20 = job_to_node[:, node_index20].to(torch.int)
        node_inputs[:, node20, 0] += 1 / 15
        node_inputs[:, node20, 2] -= 1 / 15

        node_inputs[:, 19, 0] = 0
        node_inputs[:, 19, 2] = 0

        # step2
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        y = y * gcn_masks[:, 0]

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 1]

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 2]

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 3]

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 4]

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 5]

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 6]

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 7]

        # assemble neighboring information
        x = x + y
        gcn_output = x

        # gsn
        x = torch.concat([node_inputs, gcn_output], dim=2)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)
        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)
        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)
        gsn_global_summ_extend_node = torch.concat([
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary], dim=1).to(torch.float32)
        merge_node = torch.concat([
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2).to(torch.float32)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)
        node_outputs = torch.pow(node_outputs, 5)
        node_outputs = node_outputs / torch.sum(node_outputs)
        node_outputs = torch.pow(node_outputs, 5)
        node_outputs = node_outputs / torch.sum(node_outputs)
        node_outputs = node_outputs.reshape(-1, 20)
        node_outputs2 = node_outputs
        chose = torch.sum(torch.arange(20) * node_outputs, 1)
        chose = torch.round(chose).to(torch.int)

        # step 3
        # gcn
        # modify input
        chosen_jobs = node_to_job[:, chose]

        node_index1 = (chosen_jobs * 20 + 0).to(torch.int)
        node1 = job_to_node[:, node_index1].to(torch.int)
        node_inputs[:, node1, 0] += 1 / 15
        node_inputs[:, node1, 2] -= 1 / 15

        node_index2 = (chosen_jobs * 20 + 1).to(torch.int)
        node2 = job_to_node[:, node_index2].to(torch.int)
        node_inputs[:, node2, 0] += 1 / 15
        node_inputs[:, node2, 2] -= 1 / 15

        node_index3 = (chosen_jobs * 20 + 2).to(torch.int)
        node3 = job_to_node[:, node_index3].to(torch.int)
        node_inputs[:, node3, 0] += 1 / 15
        node_inputs[:, node3, 2] -= 1 / 15

        node_index4 = (chosen_jobs * 20 + 3).to(torch.int)
        node4 = job_to_node[:, node_index4].to(torch.int)
        node_inputs[:, node4, 0] += 1 / 15
        node_inputs[:, node4, 2] -= 1 / 15

        node_index5 = (chosen_jobs * 20 + 4).to(torch.int)
        node5 = job_to_node[:, node_index5].to(torch.int)
        node_inputs[:, node5, 0] += 1 / 15
        node_inputs[:, node5, 2] -= 1 / 15

        node_index6 = (chosen_jobs * 20 + 5).to(torch.int)
        node6 = job_to_node[:, node_index6].to(torch.int)
        node_inputs[:, node6, 0] += 1 / 15
        node_inputs[:, node6, 2] -= 1 / 15

        node_index7 = (chosen_jobs * 20 + 6).to(torch.int)
        node7 = job_to_node[:, node_index7].to(torch.int)
        node_inputs[:, node7, 0] += 1 / 15
        node_inputs[:, node7, 2] -= 1 / 15

        node_index8 = (chosen_jobs * 20 + 7).to(torch.int)
        node8 = job_to_node[:, node_index8].to(torch.int)
        node_inputs[:, node8, 0] += 1 / 15
        node_inputs[:, node8, 2] -= 1 / 15

        node_index9 = (chosen_jobs * 20 + 8).to(torch.int)
        node9 = job_to_node[:, node_index9].to(torch.int)
        node_inputs[:, node9, 0] += 1 / 15
        node_inputs[:, node9, 2] -= 1 / 15

        node_index10 = (chosen_jobs * 20 + 9).to(torch.int)
        node10 = job_to_node[:, node_index10].to(torch.int)
        node_inputs[:, node10, 0] += 1 / 15
        node_inputs[:, node10, 2] -= 1 / 15

        node_index11 = (chosen_jobs * 20 + 10).to(torch.int)
        node11 = job_to_node[:, node_index11].to(torch.int)
        node_inputs[:, node11, 0] += 1 / 15
        node_inputs[:, node11, 2] -= 1 / 15

        node_index12 = (chosen_jobs * 20 + 11).to(torch.int)
        node12 = job_to_node[:, node_index12].to(torch.int)
        node_inputs[:, node12, 0] += 1 / 15
        node_inputs[:, node12, 2] -= 1 / 15

        node_index13 = (chosen_jobs * 20 + 12).to(torch.int)
        node13 = job_to_node[:, node_index13].to(torch.int)
        node_inputs[:, node13, 0] += 1 / 15
        node_inputs[:, node13, 2] -= 1 / 15

        node_index14 = (chosen_jobs * 20 + 13).to(torch.int)
        node14 = job_to_node[:, node_index14].to(torch.int)
        node_inputs[:, node14, 0] += 1 / 15
        node_inputs[:, node14, 2] -= 1 / 15

        node_index15 = (chosen_jobs * 20 + 14).to(torch.int)
        node15 = job_to_node[:, node_index15].to(torch.int)
        node_inputs[:, node15, 0] += 1 / 15
        node_inputs[:, node15, 2] -= 1 / 15

        node_index16 = (chosen_jobs * 20 + 15).to(torch.int)
        node16 = job_to_node[:, node_index16].to(torch.int)
        node_inputs[:, node16, 0] += 1 / 15
        node_inputs[:, node16, 2] -= 1 / 15

        node_index17 = (chosen_jobs * 20 + 16).to(torch.int)
        node17 = job_to_node[:, node_index17].to(torch.int)
        node_inputs[:, node17, 0] += 1 / 15
        node_inputs[:, node17, 2] -= 1 / 15

        node_index18 = (chosen_jobs * 20 + 17).to(torch.int)
        node18 = job_to_node[:, node_index18].to(torch.int)
        node_inputs[:, node18, 0] += 1 / 15
        node_inputs[:, node18, 2] -= 1 / 15

        node_index19 = (chosen_jobs * 20 + 18).to(torch.int)
        node19 = job_to_node[:, node_index19].to(torch.int)
        node_inputs[:, node19, 0] += 1 / 15
        node_inputs[:, node19, 2] -= 1 / 15

        node_index20 = (chosen_jobs * 20 + 19).to(torch.int)
        node20 = job_to_node[:, node_index20].to(torch.int)
        node_inputs[:, node20, 0] += 1 / 15
        node_inputs[:, node20, 2] -= 1 / 15

        node_inputs[:, 19, 0] = 0
        node_inputs[:, 19, 2] = 0

        # step2
        x = node_inputs

        # raise x into higher dimension
        x = self.h_gc1(x)
        x = self.act_fn(x)
        x = self.h_gc2(x)
        x = self.act_fn(x)
        x = self.h_gc3(x)
        x = self.act_fn(x)

        # -------------------------1------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 0], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g

        y = y * gcn_masks[:, 0]

        # assemble neighboring information
        x = x + y

        # -------------------------2------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 1], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 1]

        # assemble neighboring information
        x = x + y

        # -------------------------3------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 2], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 2]

        # assemble neighboring information
        x = x + y

        # -------------------------4------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 3], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 3]

        # assemble neighboring information
        x = x + y

        # -------------------------5------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 4], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 4]

        # assemble neighboring information
        x = x + y

        # -------------------------6------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 5], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 5]

        # assemble neighboring information
        x = x + y

        # -------------------------7------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 6], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 6]

        # assemble neighboring information
        x = x + y

        # -------------------------8------------------------
        # work flow: index_select -> f -> masked assemble via adj_mat -> g
        y = x

        # process the features on the nodes
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # message passing
        y = torch.matmul(gcn_mats[:, 7], y)

        # aggregate child features
        y = self.f_gc1(y)
        y = self.act_fn(y)
        y = self.f_gc2(y)
        y = self.act_fn(y)
        y = self.f_gc3(y)
        y = self.act_fn(y)

        # remove the artifact from the bias term in g
        y = y * gcn_masks[:, 7]

        # assemble neighboring information
        x = x + y
        gcn_output = x

        # gsn
        x = torch.concat([node_inputs, gcn_output], dim=2)

        # DAG level summary
        s = x
        s = self.dag_gc1(s)
        s = self.act_fn(s)
        s = self.dag_gc2(s)
        s = self.act_fn(s)
        s = self.dag_gc3(s)
        s = self.act_fn(s)

        s = torch.matmul(summ_mats, s)
        gsn_dag_summary = s

        # global level summary
        s = self.global_gc1(s)
        s = self.act_fn(s)
        s = self.global_gc2(s)
        s = self.act_fn(s)
        s = self.global_gc3(s)
        s = self.act_fn(s)
        gsn_global_summary = torch.matmul(running_dags_mat, s)

        gsn_dag_summ_extend = torch.matmul(dag_summ_backward_map, gsn_dag_summary)
        gsn_global_summ_extend_node = torch.concat([
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary,
            gsn_global_summary], dim=1)
        merge_node = torch.concat([
            node_inputs, gcn_output,
            gsn_dag_summ_extend, gsn_global_summ_extend_node], dim=2)

        y = self.fc1(merge_node)
        y = self.act_fn(y)
        y = self.fc2(y)
        y = self.act_fn(y)
        y = self.fc3(y)
        y = self.act_fn(y)
        node_outputs = self.fc4(y)

        node_outputs = node_outputs.view([-1, 1, Max_Node])

        # valid mask on node
        node_valid_mask = (node_valid_mask - 1) * 10000.0

        # apply mask
        node_outputs = node_outputs + node_valid_mask

        # do masked softmax over nodes on the graph
        node_outputs = self.softmax(node_outputs)
        node_outputs = node_outputs.reshape(-1, 20)
        node_outputs3 = node_outputs

        ret = torch.concat([node_outputs1, node_outputs2, node_outputs3], dim=1)
        print(ret)

        return ret
