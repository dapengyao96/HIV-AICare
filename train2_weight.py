#!/usr/bin/env python
# coding: utf-8


import argparse
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from tqdm import trange

import datetime
import time
import os
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
from model import Model, Model_LSTM
from env import sample_offline_data, hivEnv
from pre_model1 import pre_Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000) # epochs when fitting NN
    parser.add_argument("--n_epoch", type=int, default=2) # epochs when fitting NFQ
    parser.add_argument("--gamma", type=float, default=1.0) 
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--act_fun", type=str, default="relu")
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--hist_len", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--lstm", default=False)
    parser.add_argument("--no-lstm", default=False)
    parser.add_argument("--expectile", type=float, default=0.8) #implicit q learning 
    parser.add_argument("--n", type=int, default=400) # number of patients
    parser.add_argument("--n_timestep", type=int, default=10) # number of timesteps
    parser.add_argument("--n_interact_past_state", type=int, default=4) # number of past states interaction
    parser.add_argument("--n_action", type=int, default=5) # number of actions in action space
    parser.add_argument("--embed_dim", type=int, default=5) # dimension of action representation
    parser.add_argument("--binary_vector", default=False)
    parser.add_argument("--eval_steps", type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--n_best_action", type=int, default=5) # number of best actions for recovery
    parser.add_argument("--implicit_q", default=False)
    parser.add_argument("--no_implicit_q", default=False)
    parser.add_argument("--cat_vector", default=False)
    parser.add_argument("--no_representative_vector", default=False)
    parser.add_argument("--drug_use_time", default=False)
    parser.add_argument("--small_action_space", default=False)
    parser.add_argument("--eta", type=float, default=0.5) 
    parser.add_argument("--single_drug_time", default=False)
    return parser.parse_args()




def training_nfq(args=get_args()):
    start_time = time.time()
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    algo_name = f'NFQ'
    log_file = f'seed_{args.seed}_{t0}_{algo_name}'
    vector_name = f'st'
    if args.binary_vector:
        vector_name = f'bi'
    if args.cat_vector:
        vector_name = f'cat'
    if args.no_representative_vector:
        vector_name= f'no_representative'
    #vector_name = f'bi' if args.binary_vector else f'st'
    log_path = os.path.join(args.logdir, algo_name, vector_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer)
    logger.print(str(args))
    
    # sample offline data and load
    sample_offline_data(args.n, args.n_timestep, args.n_interact_past_state, args.n_action, args.embed_dim, args.eta, logger, args.binary_vector, args.cat_vector, args.no_representative_vector, args.small_action_space)
    y = np.load(os.path.join(logger.writer.get_logdir(),"y_0.npy"))
    kappa = np.load("kappa.npy")
    if args.eta == 0.2:
        kappa = np.load("kappa_eta0.2.npy",allow_pickle=True)
    if args.eta == 0.8:
        kappa = np.load("kappa_eta0.8.npy",allow_pickle=True)
    n = y.shape[0]
    
    
#     drug_history = np.load(os.path.join(logger.writer.get_logdir(),"drug_history_0.npy"))
#     X = np.load(os.path.join(logger.writer.get_logdir(),"X_0.npy"))
#     J = np.load(os.path.join(logger.writer.get_logdir(),"J_0.npy"))
#     sim_proj = np.load(os.path.join(logger.writer.get_logdir(),"sim_proj_0.npy"))
#     index_regimens = np.load(os.path.join(logger.writer.get_logdir(),"index_regimens_0.npy"))
#     index_kernel = np.load(os.path.join(logger.writer.get_logdir(),"index_kernel_0.npy"))
#     action_range = np.load(os.path.join(logger.writer.get_logdir(),"action_range_0.npy"))



#     state_current = [y[i,:,:J[i]-2] for i in range(n)] # y is state matrix
#     state_next = [y[i,:,1:J[i]-1] for i in range(n)]

#     action = [drug_history[i,1:J[i]-1,:] for i in range(n)]
#     covariates = [X[i,:J[i]-2,:] for i in range(n)]

#     hist_len = args.hist_len#max(J)
#     states = np.zeros((n,max(J),(hist_len+1)*y.shape[1]+X.shape[2]))
#     for i in range(n):
#         for j in range(J[i]):
#             if hist_len <= j:
#                 tmp = np.concatenate([y[i,:,k+j-hist_len].reshape(1,-1) for k in reversed(range(hist_len+1))], axis=1)
#                 states[i,j,:] = np.concatenate([tmp, X[i,0,:].reshape(1,-1)],axis=1)
#             else:
#                 tmp = np.concatenate([y[i,:,k].reshape(1,-1) for k in reversed(range(j+1))], axis=1)
#                 states[i,j,:] = np.concatenate([tmp, np.zeros((hist_len-j)*y.shape[1]).reshape(1,-1), X[i,0,:].reshape(1,-1)],axis=1)

#     state_current = [np.transpose(states[i,:J[i]-2,:]) for i in range(n)]
#     state_next = [np.transpose(states[i,1:J[i]-1,:]) for i in range(n)]


#     x_train = [np.concatenate([np.transpose(state_current[i]),action[i]],axis=1) for i in range(n)]
#     x_train = np.vstack([x_train[i] for i in range(n)])
#     x_train = torch.FloatTensor(x_train).to(args.device)

#     a_range = sim_proj # this is action space, which use the representatin based on similarity matrix


#     x_next = np.vstack([np.transpose(state_next[i]) for i in range(n)]) 

#     state_current_train = np.vstack([np.transpose(state_current[i]) for i in range(n)]) 
#     state_next_train = np.vstack([np.transpose(state_next[i]) for i in range(n)])
#     state_current_train = torch.FloatTensor(state_current_train).to(args.device)
#     state_next_train = torch.FloatTensor(state_next_train).to(args.device)


    drug_history = np.load(os.path.join(logger.writer.get_logdir(),"drug_history_0.npy"))
    X = np.load(os.path.join(logger.writer.get_logdir(),"X_0.npy"))
    J = np.load(os.path.join(logger.writer.get_logdir(),"J_0.npy"))
    T = np.load(os.path.join(logger.writer.get_logdir(),"T_0.npy"))
    sim_proj = np.load(os.path.join(logger.writer.get_logdir(),"sim_proj_0.npy"))
    index_regimens = np.load(os.path.join(logger.writer.get_logdir(),"index_regimens_0.npy"))
    index_kernel = np.load(os.path.join(logger.writer.get_logdir(),"index_kernel_0.npy"))
    action_range = np.load(os.path.join(logger.writer.get_logdir(),"action_range_0.npy"))
    action_space_name = np.load("action_space_name.npy",allow_pickle=True)
    drug_names = np.load("drug_names.npy")
    
    state_current = [y[i,:,:J[i]-2] for i in range(n)] # y is state matrix
    state_next = [y[i,:,1:J[i]-1] for i in range(n)]

    action = [drug_history[i,1:J[i]-1,:] for i in range(n)]
    covariates = [X[i,:J[i]-2,:] for i in range(n)]
    
    action_space_name_binary = np.zeros((action_space_name.shape[0],drug_names.shape[0]))
    for i in range(action_space_name.shape[0]):
        for j in range(drug_names.shape[0]):
            if sum(action_space_name[i,:] == drug_names[j]) == 1:
                action_space_name_binary[i,j] = 1

    if args.drug_use_time:
        drug_use_time = np.empty((n,max(J),action_space_name_binary.shape[1]))
        drug_use_time[:] = np.nan
        drug_use_time[:,0,:] = 0
        for i in range(n):
            for j in range(1,J[i]-1):
                binary_regimens = action_space_name_binary[index_regimens[i,:j].astype(int)-1]
                drug_use_time[i,j] = np.sum(binary_regimens * np.array(T[i,1:(j+1)]-T[i,:j]).reshape(-1,1), axis=0)
        y_ori = np.empty((n,y.shape[1]+drug_use_time.shape[2],max(J)))
        for i in range(n):
            y_ori[i] = np.concatenate([y[i],np.transpose(drug_use_time[i])],axis=0)
        y = y_ori
        
    else:
        y_ori = np.empty((n,y.shape[1]+1,max(J)))
        for i in range(n):
            y_ori[i] = np.concatenate([y[i],T[i].reshape(1,-1)],axis=0)
        y = y_ori

#     hist_len = args.hist_len#max(J)
#     states = np.zeros((n,max(J),(hist_len+1)*y.shape[1]+X.shape[2]))
#     for i in range(n):
#         for j in range(J[i]):
#             if hist_len <= j:
#                 tmp = np.concatenate([y[i,:,k+j-hist_len].reshape(1,-1) for k in reversed(range(hist_len+1))], axis=1)
#                 states[i,j,:] = np.concatenate([tmp, X[i,0,:].reshape(1,-1)],axis=1)
#             else:
#                 tmp = np.concatenate([y[i,:,k].reshape(1,-1) for k in reversed(range(j+1))], axis=1)
#                 states[i,j,:] = np.concatenate([tmp, np.zeros((hist_len-j)*y.shape[1]).reshape(1,-1), X[i,0,:].reshape(1,-1)],axis=1)

#     state_current = [np.transpose(states[i,:J[i]-2,:]) for i in range(n)]
#     state_next = [np.transpose(states[i,1:J[i]-1,:]) for i in range(n)]


#     x_train = [np.concatenate([np.transpose(state_current[i]),action[i]],axis=1) for i in range(n)]
#     x_train = np.vstack([x_train[i] for i in range(n)])
#     x_train = torch.FloatTensor(x_train).to(args.device)

#     a_range = sim_proj # this is action space, which use the representatin based on similarity matrix


#     x_next = np.vstack([np.transpose(state_next[i]) for i in range(n)]) 

#     state_current_train = np.vstack([np.transpose(state_current[i]) for i in range(n)]) 
#     state_next_train = np.vstack([np.transpose(state_next[i]) for i in range(n)])
#     state_current_train = torch.FloatTensor(state_current_train).to(args.device)
#     state_next_train = torch.FloatTensor(state_next_train).to(args.device)

    hist_len = args.hist_len#max(J)
    states = np.zeros((n,max(J),(hist_len+1)*y.shape[1]+X.shape[2]))
    for i in range(n):
        for j in range(J[i]):
            if hist_len <= j:
                tmp = np.concatenate([y[i,:,k+j-hist_len].reshape(1,-1) for k in reversed(range(hist_len+1))], axis=1)
                states[i,j,:] = np.concatenate([tmp, X[i,0,:].reshape(1,-1)],axis=1)
            else:
                tmp = np.concatenate([y[i,:,k].reshape(1,-1) for k in reversed(range(j+1))], axis=1)
                states[i,j,:] = np.concatenate([tmp, np.zeros((hist_len-j)*y.shape[1]).reshape(1,-1), X[i,0,:].reshape(1,-1)],axis=1)

    state_current = [states[i,:J[i]-2,:] for i in range(n)]
    state_next = [states[i,1:J[i]-1,:] for i in range(n)]

    if hist_len > 0:
        actions = np.zeros((n,max(J),(hist_len)*sim_proj.shape[1]))
        for i in range(n):
            for j in range(J[i]):
                if hist_len <= j:
                    actions[i,j,:] = np.concatenate([drug_history[i,k+j-hist_len+1,:].reshape(1,-1) for k in reversed(range(hist_len))], axis=1)
                else:
                    if j>0:
                        tmp = np.concatenate([drug_history[i,k+1].reshape(1,-1) for k in reversed(range(j))], axis=1)
                        actions[i,j,:] = np.concatenate([tmp, np.zeros((hist_len-j)*sim_proj.shape[1]).reshape(1,-1)],axis=1)
                    
        actions_current = [actions[i,:J[i]-2,:] for i in range(n)]
        actions_next = [actions[i,1:J[i]-1,:] for i in range(n)]
        x_train = [np.concatenate([state_current[i],actions_current[i],action[i]],axis=1) for i in range(n)]
        x_next = np.vstack([np.concatenate([state_next[i], actions_next[i]], axis=1) for i in range(n)]) 
        
        state_current = [np.concatenate([states[i,:J[i]-2,:], actions_current[i]], axis=1) for i in range(n)]
        state_next = [np.concatenate([states[i,1:J[i]-1,:], actions_next[i]], axis=1) for i in range(n)]

    else:
        x_train = [np.concatenate([state_current[i],action[i]],axis=1) for i in range(n)]
        x_next = np.vstack([state_next[i] for i in range(n)]) 

    x_train = np.vstack([x_train[i] for i in range(n)])
    x_train = torch.FloatTensor(x_train).to(args.device)

    a_range = sim_proj # this is action space, which use the representatin based on similarity matrix


#     x_train = torch.nn.functional.normalize(x_train, dim = 0)
#     x_next = torch.nn.functional.normalize(x_next, dim = 0)

    state_current_train = np.vstack([state_current[i] for i in range(n)]) 
    state_next_train = np.vstack([state_next[i] for i in range(n)])
    state_current_train = torch.FloatTensor(state_current_train).to(args.device)
    state_next_train = torch.FloatTensor(state_next_train).to(args.device)
    

    def get_best_action(net, state, a_range): #state should be np.array
        num_a = a_range.shape[0]
        num_s = state.shape[0]

        q = []
        for i in range(num_a):
            s_a = torch.FloatTensor(np.hstack((state, np.ones((num_s,1))*a_range[i,:]))).to(args.device) # state_action pair
            with torch.no_grad():
                q.append(net(s_a).cpu().numpy().reshape(-1))

        q = np.asarray(q)
        q_opt = np.max(q, axis = 0)
        a_opt = a_range[q.argmax(axis = 0),:]
        idx = q.argmax(axis = 0)
        return(q_opt, a_opt, idx) #np array


    def get_best_k_action(net, state, a_range, k): #state should be np.array
        num_a = a_range.shape[0]
        num_s = state.shape[0]
        q = []
        for i in range(num_s):
            s_a = torch.FloatTensor(np.hstack((np.ones((num_a,1))*state[i,:], a_range))).to(args.device) # state_action pair
            with torch.no_grad():
                q.append(net(s_a).cpu().numpy().reshape(-1))



        q = np.asarray(q)

        idx = np.argpartition(q.reshape(-1),-k)[-k:]
        idx = idx[np.argsort(q.reshape(-1)[idx])]
        logger.print("best action: {}".format(idx+1))
        a_opt = a_range[q.argmax(axis = 1),:]
        q_opt = q[:,idx]
        return(q_opt, a_opt,idx) #np array


#     def evaluate(net,n_interact_past_state,n_action,eval_steps,eval_episodes):
#         phi = np.load("phi.npy",allow_pickle=True)

#         #env = hivEnv(X,index_regimens,phi,kappa,n_action,n_interact_past_state)
#         env = hivEnv(X,index_kernel,action_range,index_regimens,kappa,n_action,n_interact_past_state)

#         reward_ep = []
#         for ep in range(eval_episodes):
#             np.random.seed(int(time.time()*1000)%2**32)
#             idx = np.random.randint(n)
#             obs = env.reset(idx)
#             episode_reward = 0
#             for i in range(eval_steps):
#                 y = env.get_state_history()
#                 if hist_len <= i:
#                     #state = np.concatenate([obs.reshape(1,-1), X[idx,0,:].reshape(1,-1)],axis=1)
#                     if hist_len>0:
#                         tmp = np.concatenate([y[k+i-hist_len,:].reshape(1,-1) for k in reversed(range(hist_len))], axis=1)
#                         state = np.concatenate([obs, tmp, X[idx,0,:].reshape(1,-1)],axis=1)
#                     else:
#                         state = np.concatenate([obs.reshape(1,-1), X[idx,0,:].reshape(1,-1)],axis=1)
#                 else:
#                     if i>0:
#                         tmp = np.concatenate([y[k,:].reshape(1,-1) for k in reversed(range(i))], axis=1)
#                         state = np.concatenate([obs.reshape(1,-1), tmp, np.zeros((hist_len-i)*y.shape[1]).reshape(1,-1), X[idx,0,:].reshape(1,-1)],axis=1)
#                     else:
#                         state = np.concatenate([obs, np.zeros((hist_len-i)*y.shape[1]).reshape(1,-1), X[idx,0,:].reshape(1,-1)],axis=1)
#                 _, _, action = get_best_action(net, state, a_range)
#                 #action = index_kernel[action][0].astype(int)
#                 action = action_range[action][0].astype(int)
#                 next_obs, reward, _, _ = env.step(action, np.array([0.5*(i+1)]))
#                 episode_reward += reward
#                 obs = next_obs
#             reward_ep.append(episode_reward)
#         ep_reward_mean = np.mean(reward_ep)
#         ep_reward_std = np.std(reward_ep)
#         return ep_reward_mean, ep_reward_std

    
    
    def train(net, optimizer, gamma):
        epochs = args.epochs #epochs when fitting NN
        expectile = args.expectile #implicit q learning 
        expected_return = []
        losses = np.full(epochs, np.nan)
#         reward = torch.FloatTensor(-x_next[:,0]-x_next[:,1]+x_next[:,2]).to(args.device).div(3)
        reward = ((x_next[:,1]<0)*(x_next[:,1]-0) + (x_next[:,2]<0)*(x_next[:,2]-0) + (x_next[:,3]<0)*(x_next[:,3]-0))/3
        reward = (x_next[:,0]<0)*(-100 + x_next[:,0]-0) + (x_next[:,0]>0)*(reward)
        reward = torch.FloatTensor(reward).to(args.device)
        
        q_next_best, _, _ = get_best_action(net, x_next, a_range)
        q_next_best = torch.FloatTensor(q_next_best).to(args.device)

        with torch.no_grad():
            q_target = reward + gamma * q_next_best
            q_target = q_target.reshape((len(q_target),1))

        for epoch in trange(epochs):
            q_predict = net(x_train)
            loss = F.mse_loss(q_predict, q_target) 
            optimizer.zero_grad()
            loss.backward()#retain_graph=True
            optimizer.step()
            losses[epoch] = loss.item()
            # evaluate policy
#             ep_reward_mean, ep_reward_std = evaluate(net, args.n_interact_past_state, args.n_action, args.eval_steps, args.eval_episodes)
#             expected_return.append(ep_reward_mean)
#             if ((epoch+1) % 100)==0:
#                 logger.print(f"Epoch #{epoch}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}")
#         torch.save(expected_return, os.path.join(logger.writer.get_logdir(), "expected_return.pth"))
        logger.print('loss: {}'.format(loss))
        return losses
        
        
        
    def train_iq(V_net, Q_net, V_optimizer, Q_optimizer, gamma):
        epochs = args.epochs #epochs when fitting NN
        expectile = args.expectile #implicit q learning 
        expected_return = []
        losses_v = np.full(epochs, np.nan)
        losses_q = np.full(epochs, np.nan)
#         reward = torch.FloatTensor(-x_next[:,0]-x_next[:,1]+x_next[:,2]).to(args.device).div(3)
        reward = ((x_next[:,1]<0)*(x_next[:,1]-0) + (x_next[:,2]<0)*(x_next[:,2]-0) + (x_next[:,3]<0)*(x_next[:,3]-0))/3
        reward = (x_next[:,0]<0)*(-100 + x_next[:,0]-0) + (x_next[:,0]>0)*(reward)
        reward = torch.FloatTensor(reward).to(args.device)
#         reward = reward/100

        # update V
        with torch.no_grad():
            v_target = Q_net(x_train)
        for epoch in trange(epochs):
            v_predict = V_net(state_current_train)
            diff = v_target - v_predict
            weight = torch.tensor([expectile if x >= 0 else 1-expectile for x in diff]).to(args.device)
            loss_v = torch.mean(weight * (diff**2))
            V_optimizer.zero_grad()
            loss_v.backward()#retain_graph=True
            V_optimizer.step()
            losses_v[epoch] = loss_v.item()
        # update Q
        with torch.no_grad():
            q_target = reward.reshape(-1,1) + gamma*V_net(state_next_train)
        for epoch in trange(epochs):
            q_predict = Q_net(x_train)
            diff = q_target - q_predict
            loss_q = torch.mean((diff**2))
            Q_optimizer.zero_grad()
            loss_q.backward()#retain_graph=True
#             torch.nn.utils.clip_grad_norm_(Q_net.parameters(), 5)
            Q_net.get_grad_flow(Q_net.named_parameters())
            Q_optimizer.step()
            losses_q[epoch] = loss_q.item()
        logger.print('loss_v: {}'.format(loss_v))
        logger.print('loss_q: {}'.format(loss_q))
        return losses_v, losses_q
    

    def main(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_epoch = args.n_epoch
        gamma = args.gamma
        losses = []
        if args.implicit_q:
#             inputDim = x_train.shape[1]
            V_net = Model(state_current_train.shape[1],args.hidden1,args.hidden2).to(args.device)
            Q_net = Model(x_train.shape[1],args.hidden1,args.hidden2).to(args.device)
            net = [V_net, Q_net]
            learningRate = args.lr
            V_optimizer = torch.optim.Adam(V_net.parameters(), lr=learningRate)
            Q_optimizer = torch.optim.Adam(Q_net.parameters(), lr=learningRate)
            optimizer = [V_optimizer, Q_optimizer]
            
            for epoch in range(n_epoch):
                if epoch % 1 == 0:
                    logger.print('Epoch {:4d}: '.format(epoch + 1))
#                 loss_v, loss_q = train_iq(V_net, Q_net, V_optimizer, Q_optimizer, gamma)
                loss_v, loss_q = train_iq(V_net, Q_net, V_optimizer, Q_optimizer, gamma)
                loss = [loss_v, loss_q]
                losses.append(loss)
            torch.save(Q_net.max_grads, os.path.join(logger.writer.get_logdir(), "max_grads.pth"))
            torch.save(Q_net.ave_grads, os.path.join(logger.writer.get_logdir(), "ave_grads.pth"))
            
            return Q_net, losses
            
        if args.no_implicit_q:
            net = Model(x_train.shape[1],args.hidden1,args.hidden2).to(args.device)
            learningRate = args.lr
            optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)

            for epoch in range(n_epoch):
                if epoch % 1 == 0:
                    logger.print('Epoch {:4d}: '.format(epoch + 1))
#                 loss = train(net, optimizer, gamma)
                loss = train(net, optimizer, gamma)
                losses.append(loss)
            return net,losses





    recover = np.zeros(n)
    Q_net, loss = main(args.seed)
    best_action = np.load(os.path.join(logger.writer.get_logdir(),"best_action.npy"))
    for i in range(n):
        logger.print("patient {}".format(i))
        if J[i]-3>0:
            aaa,bbb,ccc = get_best_k_action(Q_net,state_current[i][J[i]-4,:].reshape((1,state_current[0].shape[1])),a_range,args.n_best_action)
            if (sum((ccc+1) == best_action[i]) == 1):
                recover[i] = 1
            logger.print("q value: {}".format(aaa))

    torch.save(Q_net.state_dict(), os.path.join(logger.writer.get_logdir(), "net.pth"))
    logger.print("Recover {} patients".format(np.sum(recover)))
    torch.save(np.sum(recover), os.path.join(logger.writer.get_logdir(), "recover.pth"))
    logger.print("total time: {:.3f}s".format(time.time() - start_time))
    torch.save(loss, os.path.join(logger.writer.get_logdir(), "loss.pth"))
    
    
########## LSTM-NFQ

def training_lstm_nfq(args=get_args()):
    start_time = time.time()
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    algo_name = f'LSTM-NFQ'
    log_file = f'seed_{args.seed}_{t0}_{algo_name}'
    vector_name = f'st'
    if args.binary_vector:
        vector_name = f'bi'
    if args.cat_vector:
        vector_name = f'cat'
    if args.no_representative_vector:
        vector_name= f'no_representative'
    #vector_name = f'bi' if args.binary_vector else f'st'
    log_path = os.path.join(args.logdir, algo_name, vector_name,  log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer)
    logger.print(str(args))

    # sample offline data and load
    sample_offline_data(args.n, args.n_timestep, args.n_interact_past_state, args.n_action, args.embed_dim, args.eta, logger, args.binary_vector, args.cat_vector, args.no_representative_vector, args.small_action_space)
    y = np.load(os.path.join(logger.writer.get_logdir(),"y_0.npy"))
    kappa_0 = np.load("kappa.npy",allow_pickle=True)
    kappa = np.load("kappa.npy")
    if args.eta == 0.2:
        kappa = np.load("kappa_eta0.2.npy",allow_pickle=True)
    if args.eta == 0.8:
        kappa = np.load("kappa_eta0.8.npy",allow_pickle=True)
    n = y.shape[0]
    n_action = args.n_action

    drug_history = np.load(os.path.join(logger.writer.get_logdir(),"drug_history_0.npy"))
    X = np.load(os.path.join(logger.writer.get_logdir(),"X_0.npy"))
    J = np.load(os.path.join(logger.writer.get_logdir(),"J_0.npy"))
    T = np.load(os.path.join(logger.writer.get_logdir(),"T_0.npy"))
    sim_proj = np.load(os.path.join(logger.writer.get_logdir(),"sim_proj_0.npy"))
    index_regimens = np.load(os.path.join(logger.writer.get_logdir(),"index_regimens_0.npy"))
    index_kernel = np.load(os.path.join(logger.writer.get_logdir(),"index_kernel_0.npy"))
    action_range = np.load(os.path.join(logger.writer.get_logdir(),"action_range_0.npy"))
    action_space_name = np.load("action_space_name.npy",allow_pickle=True)
    drug_names = np.load("drug_names.npy")
    
    action_space_name_binary = np.zeros((action_space_name.shape[0],drug_names.shape[0]))
    for i in range(action_space_name.shape[0]):
        for j in range(drug_names.shape[0]):
            if sum(action_space_name[i,:] == drug_names[j]) == 1:
                action_space_name_binary[i,j] = 1

    if args.drug_use_time:
        drug_use_time = np.empty((n,max(J),action_space_name_binary.shape[1]))
#         drug_use_time_last = np.empty((n,max(J),action_space_name_binary.shape[1]))
        drug_use_time[:] = np.nan
#         drug_use_time_last[:] = np.nan
        drug_use_time[:,0,:] = 0
#         drug_use_time_last[:,0,:] = 0
        for i in range(n):
            for j in range(1,J[i]-1):
                binary_regimens = action_space_name_binary[index_regimens[i,:j].astype(int)-1]
                drug_use_time[i,j] = np.sum(binary_regimens * np.array(T[i,1:(j+1)]-T[i,:j]).reshape(-1,1), axis=0)
#                 for drug in range(action_space_name_binary.shape[1]):
#                     if np.sum(binary_regimens[:,drug])>0:
#                         drug_use_time_last[i,j,drug] = T[i,j] - T[i,1:(j+1)][np.where(binary_regimens[:,drug]==1)[0][-1]]
#                     else:
#                         drug_use_time_last[i,j,drug] = T[i,j]
#         y_ori = np.empty((n,y.shape[1]+drug_use_time.shape[2]*2,max(J)))
#         for i in range(n):
#             y_ori[i] = np.concatenate([y[i],np.transpose(drug_use_time[i]),np.transpose(drug_use_time_last[i])],axis=0)
        y_ori = np.empty((n,y.shape[1]+drug_use_time.shape[2],max(J)))
        for i in range(n):
            y_ori[i] = np.concatenate([y[i],np.transpose(drug_use_time[i])],axis=0)
        y = y_ori


    state_current = [y[i,:,:J[i]-2] for i in range(n)] # y is state matrix
    state_next = [y[i,:,1:J[i]-1] for i in range(n)]

    action = np.vstack([drug_history[i,1:J[i]-1,:] for i in range(n)])
    covariates = [X[i,:J[i]-2,:] for i in range(n)]


    ## add X to the state vector
    state_current = np.vstack([np.concatenate([np.transpose(state_current[i]),covariates[i]],axis=1) for i in range(n)])
    state_next = np.vstack([np.concatenate([np.transpose(state_next[i]),covariates[i]],axis=1) for i in range(n)])


    a_range = sim_proj # this is action space, which use the representatin based on similarity matrix


    hist_len = max(J)
    state_history = np.zeros((n,max(J),hist_len,y.shape[1]+X.shape[2]))
    for i in range(n):
        for j in range(J[i]):
            if hist_len <= j:
                state_history[i,j,:,:] = np.concatenate([np.transpose(y[i,:,(j-hist_len):j]), X[i,(j-hist_len):j,:]],axis=1)
            else:
                if j>0:
                    state_history[i,j,-(j):,:] = np.concatenate([np.transpose(y[i,:,:(j)]), X[i,:(j),:]],axis=1)

    state_current_history = np.concatenate([state_history[i,:J[i]-2,:,:] for i in range(n)],axis = 0)
    state_next_history = np.concatenate([state_history[i,1:J[i]-1,:,:] for i in range(n)],axis = 0)



    action_history = np.zeros((n,max(J),hist_len,action.shape[1]))
    for i in range(n):
        for j in range(J[i]):
            if hist_len <= j:
                action_history[i,j,:,:] = drug_history[i,(j-hist_len):j,:]
            else:
                if j>0:
                    action_history[i,j,-(j):,:] = drug_history[i,:(j),:]
    action_history = np.concatenate([action_history[i,:J[i]-2,:,:] for i in range(n)],axis = 0)
    

    time_history = np.zeros((n,max(J),hist_len,1))
    for i in range(n):
        for j in range(J[i]):
            if hist_len <= j:
                time_history[i,j,:,:] = T[i,(j-hist_len):j].reshape(1,-1,1)
            else:
                if j>0:
                    time_history[i,j,-(j):,:] = T[i,:(j)].reshape(1,-1,1)
                    
    
    time_history_current = np.concatenate([time_history[i,:J[i]-2,:,:] for i in range(n)],axis = 0)
    time_history_next = np.concatenate([time_history[i,1:J[i]-1,:,:] for i in range(n)],axis = 0)

    time_current = np.concatenate([T[i,:J[i]-2] for i in range(n)],axis=0).reshape(-1,1)
    time_next = np.concatenate([ T[i,1:J[i]-1] for i in range(n)],axis=0).reshape(-1,1)
    state_current = np.concatenate([state_current, time_current], axis = -1)
    state_next = np.concatenate([state_next, time_next], axis = -1)
    state_current_history = np.concatenate([state_current_history, time_history_current], axis = -1)
    state_next_history = np.concatenate([state_next_history, time_history_next], axis = -1)

    state_current = torch.FloatTensor(state_current).to(args.device)
    state_next = torch.FloatTensor(state_next).to(args.device)
    action = torch.FloatTensor(action).to(args.device)
    state_current_history = torch.FloatTensor(state_current_history).to(args.device)
    state_next_history = torch.FloatTensor(state_next_history).to(args.device)
    a_range = torch.FloatTensor(a_range).to(args.device)
    action_history = torch.FloatTensor(action_history).to(args.device)
    
    action_next_history = np.zeros((n,max(J),hist_len,action.shape[1]))
    for i in range(n):
        for j in range(J[i]):
            if hist_len <= j+1:
                action_next_history[i,j,:,:] = drug_history[i,(j+1-hist_len):(j+1),:]
            else:
                if j+1>0:
                    action_next_history[i,j,-(j+1):,:] = drug_history[i,:(j+1),:]
    action_next_history = np.concatenate([action_next_history[i,:J[i]-2,:,:] for i in range(n)],axis = 0)
    action_next_history = torch.FloatTensor(action_next_history).to(args.device)

    
#     def get_best_action(net, state, a_range, state_history, action_history): 
#         num_a = a_range.shape[0]
#         num_s = state.shape[0]

#         q = []

#         for i in range(num_a):
#             with torch.no_grad():
#                 q.append(net(state, a_range[i,:].repeat(num_s,1), state_history, action_history)[0].numpy().reshape(-1))

#         q = np.asarray(q)
#         q_opt = np.max(q, axis = 0)
#         a_opt = a_range[q.argmax(axis = 0),:]
#         return(q_opt, a_opt) #np array
    
    
    def get_best_k_action(net, state, a_range, state_history, action_history, k): 
        num_a = a_range.shape[0]
        num_s = state.shape[0]
        q = []


        for i in range(num_s):
            with torch.no_grad():
                q.append(net(state.repeat(num_a,1), a_range, state_history.repeat(num_a,1,1), action_history.repeat(num_a,1,1)).numpy())

        q = np.asarray(q)

        idx = np.argpartition(q.reshape(-1),-k)[-k:]
        idx = idx[np.argsort(q.reshape(-1)[idx])]
#         print("best action:",idx+1)
        a_opt = a_range[q.argmax(axis = 1),:]
        q_opt = q[:,idx]
        return(q_opt, a_opt,idx) #np array
    
#     def evaluate(net,n_interact_past_state,n_action,eval_steps,eval_episodes):
#         phi = np.load("phi.npy",allow_pickle=True)

#         #env = hivEnv(X,index_regimens,phi,kappa,n_action,n_interact_past_state)
#         env = hivEnv(X,index_kernel,action_range,index_regimens,kappa_0,n_action,n_interact_past_state)
        
#         reward_ep = []
#         for ep in range(eval_episodes):
#             np.random.seed(int(time.time()*1000)%2**32)
#             idx = np.random.randint(n)
#             obs = env.reset(idx)
#             episode_reward = 0
#             T = [0]
#             drug_history = np.empty((0,a_range.shape[1]))
#             for i in range(eval_steps):
#                 state = np.concatenate([obs.reshape(1,-1), X[idx,0,:].reshape(1,-1), np.array(T[-1]).reshape(1,-1)], axis=1)
#                 y = env.get_state_history()
#                 state_history = np.zeros((1,hist_len,y.shape[1]+X.shape[2]+1))
#                 action_history = np.zeros((1,hist_len,a_range.shape[1]))
#                 if hist_len <= i:
#                     state_history[0] = np.concatenate([y[(i-hist_len):i,:],X[idx,(i-hist_len):i,:],np.array(T[(i-hist_len):i]).reshape(-1,1)],axis=1)
#                     action_history[0] = drug_history[(i-hist_len):i,:]
#                 else:
#                     if i>0:
#                         state_history[0,-(i):,:] = np.concatenate([y[:-1,:],X[idx,:i,:],np.array(T[:-1]).reshape(-1,1)],axis=1)
#                         action_history[0,-(i):,:] = drug_history
#                 state_history = torch.tensor(state_history).float().to(args.device)
#                 action_history = torch.tensor(action_history).float().to(args.device)
#                 state = torch.tensor(state).float().to(args.device)
#                 _, _, action = get_best_action(net, state, a_range, state_history, action_history)
#                 drug_history = np.append(drug_history, a_range[action,:].cpu().numpy(), axis = 0)
                
#                 #action = index_kernel[action][0].astype(int)
#                 action = action_range[action][0].astype(int)
#                 next_obs, reward, _, _ = env.step(action, np.array([0.5*(i+1)]))
#                 episode_reward += reward
#                 obs = next_obs
#                 T.append(0.5*(i+1))
#             reward_ep.append(episode_reward)
#         ep_reward_mean = np.mean(reward_ep)
#         ep_reward_std = np.std(reward_ep)
#         return ep_reward_mean, ep_reward_std
    
    import torch.nn.init as init

    def pre_train(V_net, Q_net, V_optimizer, Q_optimizer, gamma, seed, val_id, k_epoch):
        epochs = args.epochs #epochs when fitting NN
        expectile = args.expectile #implicit q learning 

        losses_v = np.full(epochs, np.nan)
        losses_q = np.full(epochs, np.nan)
        losses_val_v = np.full(epochs, np.nan)
        losses_val_q = np.full(epochs, np.nan)
        
        acc = np.full(epochs, np.nan)
        acc_val = np.full(epochs, np.nan)
        
        train_id = np.setdiff1d(np.arange(state_current.shape[0]), val_id)

#         reward = 0*(state_next[:,1]<0)*(state_next[:,1]-0)
        reward = (state_next[:,0]<0)*(-100) #+ (state_next[:,0]>0)*(reward)



        V_scheduler = torch.optim.lr_scheduler.StepLR(V_optimizer, step_size=10, gamma=0.1) #reduce the lr by a factor of 0.1 every 10 epochs
        Q_scheduler = torch.optim.lr_scheduler.StepLR(Q_optimizer, step_size=10, gamma=0.1) #reduce the lr by a factor of 0.1 every 10 epochs
        #V_net.reset_params()
        #Q_net.reset_params()
        best_acc = 0
        best_loss_v = 10000000
        best_loss_q = 10000000
        saved_epoch = 0

        
        for epoch in trange(epochs):
            V_net.train()
#             idx = np.random.choice(train_id.shape[0], size = 256, replace=False)
            idx = np.arange(train_id.shape[0])
            
            with torch.no_grad():
                v_target = torch.round(Q_net(state_current[train_id][idx], action[train_id][idx], state_current_history[train_id][idx], action_history[train_id][idx]))
#                 v_target = Q_net(state_current[train_id][idx], action[train_id][idx], state_current_history[train_id][idx], action_history[train_id][idx])
#                 print("v_target:",np.unique(v_target,return_counts=True))
            
            # update V function
            v_predict = V_net(state_current[train_id][idx], action[train_id][idx], state_current_history[train_id][idx], action_history[train_id][idx])
#             for param in V_net.parameters():
#                 print("grad:",param.grad)
            diff = (v_target - v_predict)
            weight = torch.tensor([expectile if x >= 0 else 1-expectile for x in diff])
            loss_v = torch.mean(weight * (diff**2))
#             los_v = nn.MSELoss()
#             loss_v = los_v(v_predict, v_target)
            V_optimizer.zero_grad()
            loss_v.backward()
            V_optimizer.step()
#             V_scheduler.step()
            losses_v[epoch] = loss_v.item()
#             print("vloss:",loss_v)
            
            V_net.eval()
            with torch.no_grad():
                v_target = torch.round(Q_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id]))
#                 v_target = Q_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id])
                v_predict = V_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id])
                
                diff = (v_target - v_predict)
                weight = torch.tensor([expectile if x >= 0 else 1-expectile for x in diff])
                losses_val_v[epoch] = torch.mean(weight * (diff**2))
            if losses_val_v[epoch] < best_loss_v:
                saved_epoch = epoch
                V_file_name = 'V_best_1_' + str(k_epoch) + '.pth'
                torch.save(V_net.state_dict(), os.path.join(logger.writer.get_logdir(), V_file_name))
                best_loss_v = losses_val_v[epoch]
        logger.print("save epoch V: {}".format(saved_epoch))    
        V_net.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), V_file_name)))
    
        for epoch in trange(epochs):
            Q_net.train()
#             idx = np.random.choice(train_id.shape[0], size = 256, replace=False)
            idx = np.arange(train_id.shape[0])
    
            with torch.no_grad():
                q_target = reward[train_id][idx]/100 + torch.round(gamma*V_net(state_next[train_id][idx], action[train_id][idx], state_next_history[train_id][idx], action_history[train_id][idx]))
#                 q_target = reward[train_id][idx]/100 + gamma*V_net(state_next[train_id][idx], action[train_id][idx], state_next_history[train_id][idx], action_history[train_id][idx])
#                 print("q_target:",np.unique(q_target,return_counts=True))
            
            # update Q function
            
            q_predict = Q_net(state_current[train_id][idx], action[train_id][idx], state_current_history[train_id][idx], action_history[train_id][idx])
            
            q_target_unique, q_target_inverse = torch.unique(q_target, return_inverse=True)
            q_target_count = torch.bincount(q_target_inverse)
#             weight = 1/q_target_count[q_target_inverse]
            weight = q_target_count.sum()/q_target_count[q_target_inverse]
            loss_q = torch.mean(weight * (q_target - q_predict)**2)
#             los_q = nn.MSELoss()
#             loss_q = los_q(q_predict, q_target)
            Q_optimizer.zero_grad()
            loss_q.backward()
#             net.get_grad_flow(net.named_parameters())
            Q_optimizer.step()
#             Q_scheduler.step()
            losses_q[epoch] = loss_q.item()
#             print("qloss:",loss_q)
            acc[epoch] = ((q_predict == q_target)*1.).mean()
            
            
            Q_net.eval()
            with torch.no_grad():
                q_target = reward[val_id]/100 + torch.round(gamma*V_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id]))
#                 q_target = reward[val_id]/100 + gamma*V_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id])
                q_predict = Q_net(state_current[val_id], action[val_id], state_current_history[val_id], action_history[val_id])
                
                q_target_unique, q_target_inverse = torch.unique(q_target, return_inverse=True)
#                 q_target_count = torch.bincount(q_target_inverse)
#                 weight = 1/q_target_count[q_target_inverse]
                weight = q_target_count.sum()/q_target_count[q_target_inverse]
                losses_val_q[epoch] = torch.mean(weight * (q_target - q_predict)**2)
                
#                 losses_val_q[epoch] = los_q(q_predict, q_target)
                acc_val[epoch] = ((q_target == q_predict)*1.).mean()
            if losses_val_q[epoch] < best_loss_q:
                saved_epoch = epoch
                Q_file_name = 'Q_best_1_' + str(k_epoch) + '.pth'
                torch.save(Q_net.state_dict(), os.path.join(logger.writer.get_logdir(), Q_file_name))
                best_acc = acc_val[epoch]
                best_loss_q = losses_val_q[epoch]
            
#         print('loss_v:',loss_v)
#         print('loss_q:',loss_q)
        logger.print("save epoch Q: {}".format(saved_epoch))
        return losses_v,losses_q ,q_target, losses_val_v, losses_val_q, acc_val, acc
    
    def main(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_epoch = args.n_epoch
        gamma = args.gamma
        inputDim = state_current.shape[1] #dim of state_action
        outputDim = 1 #dim of Q
        learningRate = args.lr
        V_net = pre_Model(state_current.shape[1],action.shape[1],with_act=False,hidden_size=args.hidden,act_fun=args.act_fun).to(args.device)
        Q_net = pre_Model(state_current.shape[1],action.shape[1],with_act=True,hidden_size=args.hidden,act_fun=args.act_fun).to(args.device)

        V_optimizer = torch.optim.Adam(V_net.parameters(), lr=learningRate)
        Q_optimizer = torch.optim.Adam(Q_net.parameters(), lr=learningRate)

        losses_v = []
        losses_q = []
        losses_val_v = []
        losses_val_q = []
        acc = []
        acc_val = []
        n_state = state_current.shape[0]
        val_id = np.random.choice(n_state, np.round(n_state/5).astype(int), replace=False)
    
        for epoch in range(n_epoch):
            if epoch % 1 == 0:
                print('Epoch {:4d}: '.format(epoch + 1))
            if epoch > 0:
                Q_file_name = 'Q_best_1_' + str(epoch-1) + '.pth'
                Q_net.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), Q_file_name)))
                V_file_name = 'V_best_1_' + str(epoch-1) + '.pth'
                V_net.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), V_file_name)))
            tmp_loss_v, tmp_loss_q, q_target, tmp_loss_val_v, tmp_loss_val_q, tmp_acc_val, tmp_acc = pre_train(V_net, Q_net, V_optimizer, Q_optimizer, gamma, seed, val_id, epoch)
            losses_v.append(tmp_loss_v)
            losses_q.append(tmp_loss_q)
            losses_val_v.append(tmp_loss_val_v)
            losses_val_q.append(tmp_loss_val_q)
            acc.append(tmp_acc)
            acc_val.append(tmp_acc_val)
        return V_net, Q_net, losses_v, losses_q, q_target, losses_val_v, losses_val_q, acc, acc_val
    

    
    q_value1 = np.zeros((n,action_range.size))
    V_nets1, Q_nets1, losses1_v, losses1_q, q_target1, losses_val1_v, losses_val1_q, acc1, acc_val1 = main(args.seed)
    V_file_name = 'V_best_1_' + str(args.n_epoch-1) + '.pth'
    V_nets1.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), V_file_name)))
    Q_file_name = 'Q_best_1_' + str(args.n_epoch-1) + '.pth'
    Q_nets1.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), Q_file_name)))
    
    for i in range(n):
        if J[i]-3>0:
            print("patient",i)
            idx_last = np.cumsum(J-2)[i] - 2
            aaa,bbb,ccc = get_best_k_action(Q_nets1,state_current[idx_last].unsqueeze(0),a_range,state_current_history[idx_last].unsqueeze(0),action_history[idx_last].unsqueeze(0),n_action)
            aaa = np.round(aaa) * 100
            q_value1[i,ccc] = aaa
            print("q value:",aaa)

    torch.save(V_nets1.state_dict(), os.path.join(logger.writer.get_logdir(), "V_net1.pth"))
    torch.save(Q_nets1.state_dict(), os.path.join(logger.writer.get_logdir(), "Q_net1.pth"))
    torch.save(losses1_v, os.path.join(logger.writer.get_logdir(), "losses1_v.pth"))
    torch.save(losses1_q, os.path.join(logger.writer.get_logdir(), "losses1_q.pth"))
    np.save(os.path.join(logger.writer.get_logdir(), "q_value1.npy"), q_value1)
    np.save(os.path.join(logger.writer.get_logdir(), "losses_val1_v.npy"), losses_val1_v)
    np.save(os.path.join(logger.writer.get_logdir(), "losses_val1_q.npy"), losses_val1_q)
    np.save(os.path.join(logger.writer.get_logdir(), "acc1.npy"), acc1)
    np.save(os.path.join(logger.writer.get_logdir(), "acc_val1.npy"), acc_val1)
    
### second step (independent from first step)
    
    def get_best_action1(net, state, a_range, state_history, action_history): 
        num_a = a_range.shape[0]
        num_s = state.shape[0]

        q = []
        
        for i in range(num_a):
            with torch.no_grad():
                q.append(net(state, a_range[i,:].repeat(num_s,1), state_history, action_history).numpy().reshape(-1))

        q = np.asarray(q)
        q_opt = np.max(q, axis = 0)
        a_opt = a_range[q.argmax(axis = 0),:]
        return(q_opt, a_opt) #np array
    
    
    def get_best_k_action1(net, state, a_range, state_history, action_history, k): 
        num_a = a_range.shape[0]
        num_s = state.shape[0]
        q = []

        
        for i in range(num_s):
            with torch.no_grad():
                q.append(net(state.repeat(num_a,1), a_range, state_history.repeat(num_a,1,1), action_history.repeat(num_a,1,1)).numpy())

        q = np.asarray(q)
        idx = np.argpartition(q.reshape(-1),-k)[-k:]
        idx = idx[np.argsort(q.reshape(-1)[idx])]
#         print("best action:",idx+1)
        a_opt = a_range[q.argmax(axis = 1),:]
        q_opt = q[:,idx]
        return(q_opt, a_opt,idx) #np array
    
    
    cand_act = np.empty((state_current.shape[0],a_range.shape[0]))
    for i in range(state_current.shape[0]):
        aaa,bbb,ccc = get_best_k_action(Q_nets1,state_current[i].unsqueeze(0),a_range,state_current_history[i].unsqueeze(0),action_history[i].unsqueeze(0),n_action)
        aaa = np.round(aaa) * 100
        q_value = np.zeros(aaa.size)
        for j in range(q_value.size):
            q_value[ccc[j]] = (aaa.reshape(-1))[j]
        cand_act[i] = np.array([q_value[ii]  == q_value.max() for ii in range(q_value.shape[0])])
            
            

    
    
    def train(net, net1, optimizer, gamma, 
                   state_current0, state_next0,
                   state_current_history0, state_next_history0,
                   action0, action_history0, action_next_history0,
                   cand_act0,
                   scheduler, val_id, k_epoch):
        epochs = args.epochs #epochs when fitting NN
        expectile = args.expectile #implicit q learning 

        losses = np.full(epochs, np.nan)
        losses_val = np.full(epochs, np.nan)
        
        train_id = np.setdiff1d(np.arange(state_current0.shape[0]), val_id)
    
        reward = ((state_next0[:,1]<0)*(state_next0[:,1]-0) + (state_next0[:,2]<0)*(state_next0[:,2]-0) + (state_next0[:,3]<0)*(state_next0[:,3]-0))/3
        reward = (state_next0[:,0]<0)*(0 + state_next0[:,0]-0) + (state_next0[:,0]>0)*(reward)
        reward = torch.FloatTensor(reward)

        q_next_best = np.empty(state_next0.shape[0])
#         q_next_best, _ = get_best_action1(net, net1, state_next0, a_range, state_next_history0, action_next_history0)
        for i in range(state_next.shape[0]):
            q_next_best[i], _ = get_best_action1(net, state_next0[i].unsqueeze(0), a_range[np.where(cand_act0[i]==1)], state_next_history0[i].unsqueeze(0), action_next_history0[i].unsqueeze(0))
        q_next_best = torch.FloatTensor(q_next_best)

        best_loss = 1000
        with torch.no_grad():
            q_target = reward + gamma * q_next_best
            #q_target = q_target.reshape(-1,1)
        
        net.initialize()
        for epoch in trange(epochs):
            net.train()
            q_predict = net(state_current0, action0, state_current_history0, action_history0)
            diff = q_target[train_id] - q_predict[train_id]
#             for param in net.parameters():
#                 print(param.grad)
#             weight = torch.tensor([expectile if x >= 0 else 1-expectile for x in diff])
#             loss = torch.mean(weight * (diff**2))
            loss = torch.mean(diff**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses[epoch] = loss.item()
            
            net.eval()
            with torch.no_grad():
                diff_val = q_target[val_id] - q_predict[val_id]
#                 weight_val = torch.tensor([expectile if x >= 0 else 1-expectile for x in diff_val])
#                 losses_val[epoch] = torch.mean(weight_val * (diff_val**2))
                losses_val[epoch] = torch.mean(diff_val**2)
            if losses_val[epoch] < best_loss:
                saved_epoch = epoch
                file_name = 'best_2_' + str(k_epoch) + '.pth'
                torch.save(net.state_dict(), os.path.join(logger.writer.get_logdir(), file_name))
                best_loss = losses_val[epoch]

        logger.print("save epoch: {}".format(saved_epoch))
        print('loss:',loss)
        return losses, losses_val
    
    def main(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_epoch = args.n_epoch
        gamma = args.gamma
        inputDim = state_current.shape[1] #dim of state_action
        outputDim = 1 #dim of Q
        learningRate = args.lr
        net = Model_LSTM(state_current.shape[1],action.shape[1],hist_with_past_act=True,hidden_size=args.hidden,act_fun=args.act_fun)
        net.initialize()

        losses = []
        losses_val = []
        n_state = state_current.shape[0]
        val_id = np.random.choice(n_state, np.round(n_state/4).astype(int), replace=False)
    
        for epoch in range(n_epoch):
            optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
            if epoch % 1 == 0:
                print('Epoch {:4d}: '.format(epoch + 1))
            if epoch > 0:
                file_name = 'best_2_' + str(epoch-1) + '.pth'
                net.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), file_name)))
            tmp_loss, tmp_loss_val = train(net, Q_nets1, optimizer, gamma,
                                           state_current, state_next,
                                           state_current_history, state_next_history,
                                           action, action_history, action_next_history,
                                           cand_act,
                                           scheduler, val_id, epoch)
            losses.append(tmp_loss)
            losses_val.append(tmp_loss_val)
        return net, losses, losses_val
    
    
    Q_nets2, losses2, losses_val2 = main(args.seed)
    file_name = 'best_2_' + str(args.n_epoch-1) + '.pth'
    Q_nets2.load_state_dict(torch.load(os.path.join(logger.writer.get_logdir(), file_name)))
    
    q_value2 = np.zeros((n,action_range.size))
    
    best_action = np.load(os.path.join(logger.writer.get_logdir(),"best_action.npy"))
    recover = np.zeros(n)
    for i in range(n):
        if J[i]-3>0:
            print("patient",i)
            idx_last = np.cumsum(J-2)[i] - 2
#             aaa,bbb,ccc = get_best_k_action1(Q_nets2,state_current[idx_last].unsqueeze(0),a_range,state_current_history[idx_last].unsqueeze(0),action_history[idx_last].unsqueeze(0),n_action)
            aaa,bbb,ccc = get_best_k_action1(Q_nets2,state_current[idx_last].unsqueeze(0),a_range[np.where(cand_act[idx_last]==1)],state_current_history[idx_last].unsqueeze(0),action_history[idx_last].unsqueeze(0),np.minimum(np.where(cand_act[idx_last]==1)[0].size,5))
            ccc = np.where(cand_act[idx_last]==1)[0][ccc]
            q_value2[i,ccc] = aaa
            if (sum((ccc+1) == best_action[i]) == 1):
                recover[i] = 1
            print("q value:",aaa)
    
    torch.save(Q_nets2.state_dict(), os.path.join(logger.writer.get_logdir(), "net2.pth"))
    torch.save(losses2, os.path.join(logger.writer.get_logdir(), "losses2.pth"))
    torch.save(losses_val2, os.path.join(logger.writer.get_logdir(), "losses_val2.pth"))
    
    
#     best_action = np.load(os.path.join(logger.writer.get_logdir(),"best_action.npy"))
#     recover = np.zeros(n)
    
#     q_value = q_value1 + q_value2
    
#     for i in range(n):
#         if J[i]-3>0:
#             ccc = np.argsort(-q_value[i])[:args.n_best_action]
#             if (sum((ccc+1) == best_action[i]) == 1):
#                 recover[i] = 1
    
    np.save(os.path.join(logger.writer.get_logdir(), "q_value2.npy"), q_value2)
    logger.print("Recover {:.3f} patients".format(np.sum(recover)))
    torch.save(np.sum(recover), os.path.join(logger.writer.get_logdir(), "recover.pth"))
    logger.print("total time: {:.3f}s".format(time.time() - start_time))

def training(args=get_args()):
    if args.lstm:
        training_lstm_nfq()
    if args.no_lstm:
        training_nfq()


if __name__ == "__main__":
    training()
