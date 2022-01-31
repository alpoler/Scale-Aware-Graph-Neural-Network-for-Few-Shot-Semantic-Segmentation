import torch.nn as nn
import torch.nn.functional as F
import torch
from model.convgru import ConvGRU
from model.convgru import ConvGRUCell
from itertools import permutations
from model.convgru import ConvLSTMCell


# class MPNN(nn.Module):
#
#     def __init__(self, channel, no_pass, device_number):
#         super(MPNN, self).__init__()
#         self.channel = channel
#         self.no_pass = no_pass
#         self.device_number = device_number
#         self.Us = [nn.Parameter(nn.init.orthogonal_(torch.empty(size=(self.channel, self.channel), dtype=torch.float, requires_grad=True,device='cuda:{}'.format(self.device_number))))
#             for i in range(self.no_pass)]
#         self.convgru = ConvGRUCell(256,256,3)
#
#     # H is assumed to be size of b* |V| x (W.H) x C , where |V| is number of nodes or resolution
#
#
#     def forward(self,nodes):
#         no_nodes = len(nodes)
#         b, c, h, w = nodes[0].shape
#         messages = [[torch.zeros(size=(b, h * w, c), dtype=torch.float32, requires_grad=False,
#                                  device='cuda:{}'.format(self.device_number)) for i in range(no_nodes - 1)] for j in range(no_nodes)]
#         for message_round in range(self.no_pass):
#             for i, edge in enumerate(permutations(range(no_nodes),2)):
#                 con = nodes[edge[0]].permute(0, 2, 3, 1).reshape(b, h*w, c) @  self.Us[message_round].unsqueeze(0) @ nodes[edge[1]].permute(0, 2, 3, 1).reshape(b, h*w, c).transpose(-1, -2)
#                 mes = torch.softmax(con, dim=-1) @ (nodes[edge[0]].permute(0, 2, 3, 1).reshape(b, h*w, c) + nodes[edge[1]].permute(0, 2, 3, 1).reshape(b, h*w, c))
#                 messages[edge[0]][(i % (no_nodes-1))] = mes
#             for i in range(len(nodes)):
#                 nodes[i] = self.convgru(sum(messages[i]).permute(0, 2, 1).reshape(b, c, h, w), nodes[i])
#         return nodes

class MPNN(nn.Module):

    def __init__(self, channel, no_pass, device_number):
        super(MPNN, self).__init__()
        self.channel = channel
        self.no_pass = no_pass
        self.device_number = device_number
        self.theta = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.phi = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.g = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.convgru = ConvGRUCell(256, 256, 3)
        self.convlstm = ConvLSTMCell(256, 256, 3, bias=True)

    # H is assumed to be size of b* |V| x (W.H) x C , where |V| is number of nodes or resolution

    def forward(self, nodes):
        no_nodes = len(nodes)
        b, c, h, w = nodes[0].shape

        for message_round in range(self.no_pass):
            # Message generation
            messages = [[] for i in range(no_nodes)]
            for i, edge in enumerate(permutations(range(no_nodes), 2)):
                e1 = nodes[edge[0]]
                e2 = nodes[edge[1]]
                theta_e1 = self.theta(e1).view(b, 256, -1)
                theta_e1 = theta_e1.permute(0, 2, 1)
                # theta_e2 = self.phi(e2).view(b, 256, -1)
                e2_to_e1 = torch.matmul(theta_e1, e2.view(b, 256, -1))
                atn_e2_to_e1 = torch.softmax(e2_to_e1, dim=-1)
                # atn_e1_to_e2 = torch.softmax(e2_to_e1, dim=-2)
                mes_e2_to_e1 = torch.matmul(atn_e2_to_e1,
                                            e2.view(b, 256, -1).permute(0, 2, 1) + e1.view(b, 256, -1).permute(0, 2, 1))
                mes_e2_to_e1 = mes_e2_to_e1.permute(0, 2, 1).view(b, c, h, w)
                messages[edge[0]].append(mes_e2_to_e1)
            # Message update
            for j in range(no_nodes):
                nodes[j] = self.convgru(sum(messages[j]), nodes[j])

        return nodes


class MPNN2(nn.Module):

    def __init__(self, channel, no_pass, device_number):
        super(MPNN2, self).__init__()
        self.channel = channel
        self.no_pass = no_pass
        self.device_number = device_number
        self.theta = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.phi = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.g = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.h = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        self.j = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(1,1),stride=(1,1),padding=0,bias=False)
        # self.g = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.convgru = ConvGRUCell(256, 256, 3)
        # self.convlstm = ConvLSTMCell(256, 256, (3, 3), bias=True)

    # H is assumed to be size of b* |V| x (W.H) x C , where |V| is number of nodes or resolution

    def forward(self, nodes):
        no_nodes = len(nodes)
        b, c, h, w = nodes[0].shape
        node_idx = [i for i in range(no_nodes)]
        # cur_state = torch.zeros(b, 256, h, w, device=self.device_number)
        for message_round in range(self.no_pass):
            # Message generation
            messages = []
            across_channels = torch.cat([nodes[i].view(b, c, -1) for i in range(no_nodes)], 1)
            for cur_node in node_idx:
                across_resolutions = torch.cat(nodes[0:cur_node] + nodes[cur_node + 1:], dim=2)
                chn_att = torch.matmul(nodes[cur_node].view(b, c, -1), across_channels.permute(0, 2, 1))
                chn_att = torch.softmax(chn_att, -1)
                chn_att = torch.matmul(chn_att, across_channels)
                pos_att = torch.matmul(self.phi(nodes[cur_node]).view(b, c, -1).permute(0, 2, 1),
                                       self.theta(across_resolutions).view(b, c, -1))
                pos_att = torch.softmax(pos_att, -1)
                pos_att = torch.matmul(pos_att, self.g(across_resolutions).view(b, c, -1).permute(0, 2, 1) + nodes[cur_node].view(b, c, -1).permute(0, 2, 1).repeat(1, 2, 1))
                messages.append(self.h(chn_att.view(b, c, h, w)) + self.j(pos_att.permute(0,2,1).view(b,c,h,w)))

            # Message update
            for j in range(no_nodes):
                # nodes[j], cur_state = self.convlstm(messages[j].permute(0, 2, 1).view(b, c, h, w),
                #                                     [nodes[j], cur_state])
                nodes[j] = self.convgru(messages[j], nodes[j])

        return nodes

    # def forward(self, H):
    #     b, v, c, h, w = H.shape
    #     for message_round in range(self.no_pass - 1):
    #         H = H.permute(0, 1, 3, 4, 2).reshape(b, v, h * w, c)
    #         all_edges = H.unsqueeze(2) @ self.Us[message_round][0] @ H.permute(0, 1, 3, 2).unsqueeze(1)
    #         index_of_self_loop = [i for i in range(H.shape[1])]
    #         all_edges = F.softmax(all_edges, dim=-1)
    #         # mask = torch.ones_like(all_edges,device='cuda:{}'.format(self.device_number),requires_grad=False)
    #         # mask[:,index_of_self_loop,index_of_self_loop,:,:] = torch.tensor(0.0,device="cuda:{}".format(self.device_number),requires_grad=False)
    #         # all_edges = mask * all_edges
    #         all_edges[:,index_of_self_loop,index_of_self_loop,:,:] = torch.tensor(0.0, device="cuda:{}".format(self.device_number),requires_grad=False)
    #         all_g = all_edges @ (H.unsqueeze(1) + H.unsqueeze(2))
    #         all_g = torch.mean(all_g, dim=2)
    #         all_g = all_g.reshape(b, -1, h, w, self.channel).permute(0, 1, 4, 2, 3)
    #         H = H.reshape(b, -1, h, w, self.channel).permute(0, 1, 4, 2, 3)
    #         H = self.convgru(all_g.reshape(b * v, self.channel, h, w), H.reshape(b * v, self.channel, h, w))
    #         H = H.view(b, v, self.channel, h, w)
    #     return H


class MPNN3(nn.Module):

    def __init__(self, channel, no_pass, device_number):
        super(MPNN3, self).__init__()
        self.channel = channel
        self.no_pass = no_pass
        self.device_number = device_number
        self.qconv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.keyconv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.valconv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)

        # self.g = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.convgru = ConvGRUCell(256, 256, 1)
        # self.convlstm = ConvLSTMCell(256, 256, (3, 3), bias=True)

    # H is assumed to be size of b* |V| x (W.H) x C , where |V| is number of nodes or resolution

    def forward(self, nodes):
        no_nodes = len(nodes)
        b, c, h, w = nodes[0].shape
        node_idx = [i for i in range(no_nodes)]
        # cur_state = torch.zeros(b, 256, h, w, device=self.device_number)
        for message_round in range(self.no_pass):
            # Message generation
            for cur_node in node_idx:

                qfeatures = self.qconv(tot_features)
                kfeatures = self.keyconv(tot_features)
                vfeatures = self.valconv(tot_features)
                messages = torch.matmul(torch.softmax(
                    torch.matmul(qfeatures.view(b, c, 3 * h * w).permute(0, 2, 1), kfeatures.view(b, c, 3 * h * w)),
                    -1),
                    vfeatures.view(b, c, 3 * h * w).permute(0, 2, 1))
                messages = messages.permute(0, 2, 1).view(b, c, 3 * h, w)
                # Message update
                for j in range(no_nodes):
                    # nodes[j], cur_state = self.convlstm(messages[j].permute(0, 2, 1).view(b, c, h, w),
                    #                                     [nodes[j], cur_state])
                    nodes[j] = self.convgru(messages[:, :, (j) * h:(j + 1) * h, :], nodes[j])

        return nodes


if __name__ == "__main__":
    H = torch.randint(low=0, high=4, size=[2, 2, 2, 4, 4], device='cuda:3', dtype=torch.float)
    # first_dim => batch | second_dim => node_number | third_dim => channel | fourth_dim => height | fiveth_dim => width
    model = MPNN(channel=2, no_pass=2, device_number=3)
    model.to("cuda:3")
    next_H = model(H)
