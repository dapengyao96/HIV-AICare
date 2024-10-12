import torch
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size=1):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size,hidden1)
#         self.l3 = torch.nn.Linear(hidden1,output_size)
        self.l2 = torch.nn.Linear(hidden1,hidden2)
        self.l3 = torch.nn.Linear(hidden2,hidden2)
        self.l4 = torch.nn.Linear(hidden2,output_size)
        self.relu = torch.nn.ReLU()

        self.ave_grads = np.empty((0,4), float)
        self.max_grads = np.empty((0,4), float)
        self.layers = []
        
        torch.nn.init.zeros_(self.l4.weight)
        torch.nn.init.zeros_(self.l4.bias)
        
        self.leaky = nn.LeakyReLU(0.05)
        
    def forward(self, input_seq):

        x = self.l1(input_seq)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
#         x = self.l3(x)
#         x = self.relu(x)
#         x = self.l3(x)
#         x = self.relu(x)
#         x = self.l3(x)
#         x = self.relu(x)
        x = self.l4(x)
        
#         x = self.l1(input_seq)
#         x = self.leaky(x)
#         x = self.l2(x)
#         x = self.leaky(x)
#         x = self.l3(x)
#         x = self.leaky(x)
#         x = self.l4(x)
        return x

    def get_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        self.ave_grads = np.vstack([self.ave_grads, ave_grads])
        self.max_grads = np.vstack([self.max_grads, max_grads])
        self.layers = layers
    
class Model_LSTM(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False,
                 hidden_size=128,
                 act_fun="relu"):
        super(Model_LSTM, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        
        mem_pre_lstm_hid_sizes=(hidden_size,)
        mem_lstm_hid_sizes=(hidden_size,)
        mem_after_lstm_hid_size=(hidden_size,)
        cur_feature_hid_sizes=(hidden_size,)
        post_comb_hid_sizes=(hidden_size,)
        
        if act_fun == "relu":
            act_fun = nn.ReLU
        if act_fun == "elu":
            act_fun = nn.ELU
        if act_fun == "softplus":
            act_fun = nn.Softplus
        
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         act_fun()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           act_fun()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        act_fun()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          act_fun()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]
#         torch.nn.init.zeros_(self.post_combined_layers[-2].weight)
#         torch.nn.init.zeros_(self.post_combined_layers[-2].bias)
#         self.initialize()

    def forward(self, obs, act, hist_obs, hist_act):
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = x[:,-1,:].squeeze(1)
        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1)#, extracted_memory
    
    def initialize(self):
        torch.nn.init.zeros_(self.post_combined_layers[-2].weight)
        torch.nn.init.zeros_(self.post_combined_layers[-2].bias)