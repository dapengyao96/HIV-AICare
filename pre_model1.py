import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, input_size, output_size, temperature=1.0):
        super(GumbelSoftmaxLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.temperature = temperature

    def forward(self, x):
        logits = self.linear(x)
        gumbel_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-20) + 1e-20)
        logits = (logits + gumbel_noise) / self.temperature
        return F.softmax(logits, dim=-1)
    
def softargmax(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result
    
class pre_Model(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(4,),
                 mem_lstm_hid_sizes=(4,),
                 mem_after_lstm_hid_size=(4,),
                 cur_feature_hid_sizes=(4,),
                 post_comb_hid_sizes=(4,),
                 with_act=True,
                 hidden_size=4,
                 act_fun="relu"):
        super(pre_Model, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.with_act = with_act
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
        
        self.q_output_set = torch.FloatTensor([0,-100,-200])#,-200,-250,-400])
        
        self.softmax = nn.Softmax(dim=1)
        #self.gs = GumbelSoftmaxLayer(3, 3, temperature=1.0)
        
        self.ave_grads = np.empty((0,7), float)
        self.max_grads = np.empty((0,7), float)
        
        self.layers = []
        
#         self.bn = nn.BatchNorm1d(4)
        # Memory
        #    Pre-LSTM
        mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
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
#         self.mem_after_lstm_layer_size = [mem_pre_lstm_layer_size[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           act_fun()]

        # Current Feature Extraction
        if self.with_act:
            cur_feature_layer_size = [obs_dim + act_dim] + [1]#list(cur_feature_hid_sizes)
        else:
            cur_feature_layer_size = [obs_dim] + [1]#list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        act_fun()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]#[self.q_output_set.size()[0]]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          act_fun()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]
        
#         torch.nn.init.xavier_normal_(self.post_combined_layers[-2].weight)
#         self.post_combined_layers[-2].weight.data.fill_(0.0001)
        self.post_combined_layers[-2].bias.data.fill_(-2)
#         self.post_combined_layers[-2].bias = nn.Parameter(torch.tensor([1.,0.,0.]))
    def forward(self, obs, act, hist_obs, hist_act):#, hist_act, hist_seg_len):
        #
        x = torch.cat([hist_obs, hist_act], dim=-1)
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
        #    History output mask to reduce disturbance cased by none history memory
#         hist_out = torch.gather(x, 1,
#                                 (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
#                                     1).long()).squeeze(1)
        hist_out = x[:,-1,:].squeeze(1)
        # Current Feature Extraction
        if self.with_act:
            x = torch.cat([obs, act], dim=-1)
        else:
            x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)

        # squeeze(x, -1) : critical to ensure q has right shape.
#         print(x[:10])

        x = torch.sigmoid(x) * (-2)

#         print(x[0])
#         x = torch.clamp(x,0,2)
#         x = torch.round(x)

#         prob = torch.sigmoid(x)
#         x = self.softmax(x)

#         x = softargmax(x) * (-100)
#         print(x)

#         prob = torch.cat((prob.view(-1,1),(1-prob).view(-1,1)),dim=1)
#         prob = prob/prob.sum(axis=1).view(-1,1)

#         print(prob.grad)

#         x = torch.round(x*2)*(-1)

#         x = torch.matmul(self.softmax(x/.01), torch.arange(3).float()) * (-100)
#         x = torch.matmul(torch.round(self.softmax(prob/.01)), torch.arange(3).float()) * (-100)
        
#         x = torch.round(prob*(self.q_output_set.size()[0]-1))/(self.q_output_set.size()[0]-1)*(self.q_output_set.min() - self.q_output_set.max()) + self.q_output_set.max()
        
        return torch.squeeze(x, -1)#, prob#, extracted_memory
    
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
        
    def reset_params(self):
#         self.post_combined_layers[-2].weight.data.fill_(0.01)
        self.post_combined_layers[-2].bias.data.fill_(-2)
#         self.post_combined_layers[-2].weight.data.fill_(1)
#         self.post_combined_layers[-2].bias.data.fill_(1)
#         init.xavier_normal_(self.post_combined_layers[-2].weight)
