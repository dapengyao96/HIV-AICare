
import numpy as np
import gym
from gym import Env, spaces
from scipy import linalg as LA
import os

class hivEnv(Env):
    def __init__(self,X,index_kernel,action_range,index_regimens,kappa,n_interact_past_state,action_space_name_binary):
        super(hivEnv, self).__init__()
        
        self.set_seed(2)
        
        self.state_history = []
        self.action_history = []
        
        # Define observation space
        self.observation_shape = (3,)
        self.observation_upper_limit = 1000
        self.observation_lower_limit = -1000
        self.observation_space = spaces.Box(low = np.ones(self.observation_shape) * self.observation_lower_limit,
                                           high = np.ones(self.observation_shape) * self.observation_upper_limit,
                                           dtype = np.float16)
        
        # Define action space
        self.action_space = spaces.Discrete(445,)
        
        self.n_episode = 0
        self.S = 3 # dim of covariates
        self.M = 4 # dim of states
        
        self.X = X
        self.index_regimens = index_regimens
        # drug used in kernel regression
        self.index_kernel = index_kernel
        self.D = self.index_kernel.size
        self.action_range = action_range
        self.kappa = kappa

        
#         self.phi = phi
#         self.phi = np.concatenate([phi[:,self.index_kernel_idx.astype(int)],phi[:,self.index_kernel_idx.astype(int)+329]],axis=1)
        
        self.phi = np.zeros((self.M,2*self.D))
        for m in range(self.M):
            if m == 0:
#                 self.phi[m,:] =  np.random.multivariate_normal(np.tile(np.concatenate((np.repeat([2,-1,-1],int((self.D-2)/3))-0.2, [0,0])), 2) + .6, 0.5*np.identity(2*self.D))
                self.phi[m,:self.D] =  np.random.multivariate_normal(np.concatenate((np.repeat([2,-1,-1],int((self.D-2)/3))-0.2, [0,0])) + .6, 0.5*np.identity(self.D))
                self.phi[m,self.D:] = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
                self.phi[m,1] -= 3
            else:
                self.phi[m,:] =  np.random.multivariate_normal(np.zeros(self.D*2), np.identity(self.D*2))
#             self.phi[m,:] =  np.random.multivariate_normal(np.tile(np.repeat([4,3],int(self.D/2))-3.55, 2), 0*np.identity(self.D*2))
            
#         for m in range(self.M):
#             for d in range(2*self.D):
#                 tmp_md = np.random.uniform(0,1)
#                 if tmp_md < 0.5:
#                     self.phi[m,d] = np.random.normal(5,.1,1)
#                 else:
#                     self.phi[m,d] = np.random.normal(-5,1,1)
        self.phi = 5*self.phi#10*self.phi
#         self.phi = np.clip(self.phi, -10, 10)
#         self.phi += 20*np.sign(self.phi)
#         print(self.phi[0])
        self.action_space_name_binary = action_space_name_binary
        self.N = action_space_name_binary.shape[1] #number of single drugs
        
        # drug toxicity
#         self.delta = np.zeros((self.M, self.kappa.shape[0]))
#         for m in range(self.M):
#             self.delta[m,:] = np.random.multivariate_normal(np.zeros(self.kappa.shape[0]), np.identity(self.kappa.shape[0]))
#         self.delta[:2,:] = 1#np.abs(self.delta[:2,:])
#         self.delta[2,:] = -1#-np.abs(self.delta[2,:])
#         self.delta = 2*self.delta
        
        self.delta = np.zeros((self.M, self.N))
        for m in range(self.M):
            self.delta[m,:] = np.random.multivariate_normal(np.zeros(self.N), np.identity(self.N))
#         self.delta[:2,:] = np.abs(self.delta[:2,:])
#         self.delta[2,:] = -np.abs(self.delta[2,:])
        self.delta = 0.2*self.delta#0.5*self.delta
#         self.delta[0] = 0.2*self.delta[0]
#         self.delta[1:] = 5*self.delta[1:]
#         
        # truth of interaction (with past states)
        self.F = 3 # number of past states interacted
        self.psi = np.zeros((self.M, self.F, self.D))
        for m in range(self.M):
            for f in range(self.F):
                if m == 0:
                    self.psi[m,f,:] = np.random.multivariate_normal(np.zeros(self.D) + 1, .1*np.identity(self.D))
#                     self.psi[m,f,:] = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
#                     self.psi[m,f,:] = np.random.multivariate_normal(np.concatenate((np.repeat([2,-1,-1],int((self.D-2)/3))-0.2, [0,0])), 0.1*np.identity(self.D))
                else:
                    self.psi[m,f,:] = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
        #self.psi = .1*np.ones((self.M, self.F, self.D))        
#         self.psi[:,:,1] = 20
        self.psi = 1.8*self.psi#5*self.psi
#         self.psi[0] = 1*self.psi[0]
#         self.psi[1:] = 1*self.psi[1:]
        
        
        # truth of main effect
        self.gamma = np.zeros((self.M,self.D))
        for m in range(self.M):
            if m == 0:
                self.gamma[m] = np.random.multivariate_normal(np.zeros(self.D) - .3, 0.1*np.identity(self.D))
            else:
                self.gamma[m] = np.random.multivariate_normal(np.zeros(self.D), np.identity(self.D))
#             self.gamma[m] =  np.random.multivariate_normal(np.repeat([4,3],int(self.D/2))-3.55, 0*np.identity(self.D))
#             self.gamma[m] =  np.random.multivariate_normal(np.concatenate((np.repeat([2,-1,-1],int((self.D-2)/3))-0.2, [0,0])), 0*np.identity(self.D))
    
                  
#         self.gamma = -np.abs(self.gamma)
#         self.gamma[2] = -self.gamma[2]
        self.gamma = 5*self.gamma#10*self.gamma
#         self.gamma = np.clip(self.gamma, -10, 10)
#         self.gamma += 20*np.sign(self.gamma)
        
        
    def set_seed(self, seed):
        seed_num = seed
        np.random.seed(seed_num)
        
    def reset(self,idx = None):
        if idx is None:
            idx = np.random.randint(200)
        self.n_episode = 0
        self.X_id = self.X[idx,:,:]
        #self.state_history = np.array([])
        self.action_history = []
        self.action_history = np.array(self.action_history)
        self.set_seed(idx)
        initial_state = np.random.multivariate_normal(np.zeros(self.M), np.identity(self.M)).reshape(1,-1)
        self.state_history = initial_state
        self.T_id = [] #time
        self.index_regimens_id = self.index_regimens[idx,:]
        return initial_state


    def step(self, action, time, evaluate=False, seed = 1):
        done = 0
        if evaluate == False:
            seed = self.state_history.size
            self.set_seed(seed)
        else:
            seed = seed
            self.set_seed(seed)
        
        self.action_history = np.append(self.action_history,action)
        self.index_regimens_id = self.action_history

        action = action - 1

        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        self.n_episode += 1

        self.H = self.kappa[action,self.index_kernel.astype(int)-1]
        self.H = self.H/np.sum(self.H)
        self.XH = np.zeros(( (self.S-1)*self.D, 1 ))
        self.XH[:self.D] = self.X_id[0,1] * self.H.reshape((-1,1))
        self.XH[self.D:] = self.X_id[0,2] * self.H.reshape((-1,1))

        
        state_pre = np.zeros((self.M,self.F))
        if self.state_history.shape[0] < (self.F+1):
            state_pre[:,-(self.state_history.shape[0]):] = np.transpose(self.state_history)[:,:].reshape((self.M,-1))
        else:
            state_pre = np.transpose(self.state_history)[:,-self.F:]

        #drug toxicity
        self.T_id.append(time)
        drug_use_time = np.zeros(self.N)
        drug_use_time_last = np.zeros(self.N) # time to now of each drug used last time 
        if len(self.T_id)>1:
            binary_regimens_id = self.action_space_name_binary[self.index_regimens_id[:len(self.T_id)-1].astype(int)-1]
            drug_use_time = np.sum(binary_regimens_id * (np.array(self.T_id)[1:]-np.array(self.T_id)[:-1]).reshape(-1,1), axis=0)
            for drug in range(self.N):
                if np.sum(binary_regimens_id[:,drug])>0:
                    drug_use_time_last[drug] = 0#self.T_id[-1] - self.T_id[1:][np.where(binary_regimens_id[:,drug]==1)[0][-1]]
                else:
                    drug_use_time_last[drug] = 0#self.T_id[-1]
        tox = self.delta.dot(drug_use_time*np.exp(-drug_use_time_last))
        
#         drug_use_time = self.T_id[-1] - self.T_id
#         tox = np.empty((self.M, len(self.T_id)))
#         tox[:] = np.nan
#         for m in range(self.M):
#             for i in range(len(self.T_id)):
#                 tox[m,i] = np.exp(-drug_use_time[i])*self.delta[m,self.index_regimens_id[i].astype(int)-1]
    
    
        state = np.zeros(self.M)
        for m in range(self.M):
            mu_im = self.phi[m,:].dot(self.XH) 
            if self.F != 0:
                mu_im += self.fun(state_pre,m).dot(self.psi[m,:,:]).dot(self.H)#
                mu_im += np.sum(tox[m])
                mu_im += self.gamma[m].dot(self.H) + 0*np.sign(self.gamma[m].dot(self.H))
#                 mu_im += 10*np.sign(mu_im)
            state[m] = np.random.normal(mu_im,.1,1)
#             print("state:",state[m])
#             if m == 0:
#                 print("first part:",self.phi[m,:].dot(self.XH))
#                 print("second part:",self.fun(state_pre,m).dot(self.psi[m,:,:]).dot(self.H))
#                 print("third part:",np.sum(tox[m]))
#                 print("forth part:",self.gamma[m].dot(self.H))

        
        self.state_history = np.vstack((self.state_history, state)) if self.state_history.size else state
        reward = self.get_reward(state)
        info = {}
        toxx = tox[0]+tox[1]#-tox[2]
        return state, toxx, reward, done, info
    
    def get_reward(self, state):
        '''
        get reward of state = (depression, viral load, and estimated glomerular filtration rate(eGFR) )
        reward = (-state[0]-state[1]+state[2])/3
        '''
        reward = ((state[1]<0)*(state[1]-0) + (state[2]<0)*(state[2]-0) + (state[3]<0)*(state[3]-0))/3
        reward = (state[0]<0)*(-100 + state[0]) + (state[0]>0)*(reward)
        
        return reward

    def get_state_history(self):
        return self.state_history
    
    def get_opt_action(self):
        '''
        should be run before 'step' to see the optimal action at this step
        '''
        self.index_regimens_id = self.action_history
        
        state_pre = np.zeros((self.M,self.F))
        if self.state_history.shape[0] < (self.F+1):
            state_pre[:,-(self.state_history.shape[0]):] = np.transpose(self.state_history).reshape((self.M,-1))
        else:
            state_pre = np.transpose(self.state_history)[:,-self.F:]
        #toxicity
        if len(self.T_id)>0:
            T_id = np.append(self.T_id, self.T_id[-1]+0.5)
        else:
            T_id = [0]
        
        
        #### return = sum of 2 rewards
        reward = np.zeros((self.action_range.size,self.action_range.size))
        states = np.zeros((self.action_range.size, self.M))
        
        
        for i in range(self.action_range.size):
            H = self.kappa[self.action_range[i].astype(int)-1,self.index_kernel.astype(int)-1]
            H = H/np.sum(H)
            XH = np.zeros(( (self.S-1)*self.D, 1 ))
            XH[:self.D] = self.X_id[0,1] * H.reshape((-1,1))
            XH[self.D:] = self.X_id[0,2] * H.reshape((-1,1))
            
            drug_use_time = np.zeros(self.N)
            drug_use_time_last = np.zeros(self.N) # time to now of each drug used last time 
            if len(self.T_id)>1:
                index_regimens_id = self.index_regimens_id[:len(T_id)-1].astype(int)
                binary_regimens_id = self.action_space_name_binary[index_regimens_id-1]
                drug_use_time = np.sum(binary_regimens_id * (np.array(T_id)[1:]-np.array(T_id)[:-1]).reshape(-1,1), axis=0)
                for drug in range(self.N):
                    if np.sum(binary_regimens_id[:,drug])>0:
                        drug_use_time_last[drug] = 0#T_id[-1] - T_id[1:][np.where(binary_regimens_id[:,drug]==1)[0][-1]]
                    else:
                        drug_use_time_last[drug] = 0#T_id[-1]
            tox = np.empty((self.M, self.action_range.size))
            tox[:] = np.nan
            tox[:,i] = self.delta.dot(drug_use_time*np.exp(-drug_use_time_last))
        
        
            
            for m in range(self.M):
#                 tox[m,-1,i] = np.exp(-drug_use_time[-1])*self.delta[m,self.action_range[i].astype(int)-1]
                states[i,m] = self.phi[m,:].dot(XH)
                if self.F != 0:
                    states[i,m] += self.fun(state_pre,m).dot(self.psi[m,:,:]).dot(H)#
#                     states[i,m] += np.sum(tox[m,:,i])
                    states[i,m] += tox[m,i]
                    states[i,m] += self.gamma[m].dot(H) 
#                     states[i,m] += 10*np.sign(states[i,m])
          
        reward1 = np.apply_along_axis(self.get_reward,1,states)
        for k in range(self.action_range.size):
            state_history_k = np.vstack((self.state_history, states[k,:])) if self.state_history.size else states[k,:]
            state_pre = np.zeros((self.M,self.F))
            if state_history_k.shape[0] < (self.F+1):
                state_pre[:,-(state_history_k.shape[0]):] = np.transpose(state_history_k).reshape((self.M,-1))
            else:
                state_pre = np.transpose(state_history_k)[:,-self.F:]
            states_next = np.zeros((self.action_range.size, self.M))
            
            T_id_k = np.append(T_id,T_id[-1]+0.5)

            
            
            for j in range(self.action_range.size):
                H = self.kappa[self.action_range[j].astype(int)-1,self.index_kernel.astype(int)-1]
                H = H/np.sum(H)
                XH = np.zeros(( (self.S-1)*self.D, 1 ))
                XH[:self.D] = self.X_id[0,1] * H.reshape((-1,1))
                XH[self.D:] = self.X_id[0,2] * H.reshape((-1,1))
                
                drug_use_time = np.zeros(self.N)
                drug_use_time_last = np.zeros(self.N) # time to now of each drug used last time 
                if len(T_id)>1:
                    index_regimens_id = np.append(self.index_regimens_id[:len(T_id)-1].astype(int),self.action_range[k].astype(int))
                else:
                    index_regimens_id = np.array([self.action_range[k].astype(int)])
                binary_regimens_id = self.action_space_name_binary[index_regimens_id-1]
                drug_use_time = np.sum(binary_regimens_id * (np.array(T_id_k)[1:]-np.array(T_id_k)[:-1]).reshape(-1,1), axis=0)
                for drug in range(self.N):
                    if np.sum(binary_regimens_id[:,drug])>0:
                        drug_use_time_last[drug] = 0#T_id_k[-1] - T_id_k[1:][np.where(binary_regimens_id[:,drug]==1)[0][-1]]
                    else:
                        drug_use_time_last[drug] = 0#T_id_k[-1]
                toxx = np.empty((self.M, self.action_range.size))
                toxx[:] = np.nan
                toxx[:,j] = self.delta.dot(drug_use_time*np.exp(-drug_use_time_last))
        
                for m in range(self.M):
                    states_next[j,m] = self.phi[m,:].dot(XH)
#                     toxx[m,-1,j] = np.exp(-drug_use_time[-1])*self.delta[m,self.action_range[j].astype(int)-1]
                    if self.F != 0:
                        states_next[j,m] += self.fun(state_pre,m).dot(self.psi[m,:,:]).dot(H)#
#                         states_next[j,m] += np.sum(toxx[m,:,j])
                        states_next[j,m] += toxx[m,j]
                        states_next[j,m] += self.gamma[m].dot(H)
#                         states_next[j,m] += 10*np.sign(states_next[j,m])
                
#                 if k == 6:
#                     if j == 6:
# #                         print("state_pre",state_pre[0])
#                         print("first part", self.phi[1,:].dot(XH))
#                         print("second part",self.fun(state_pre,1).dot(self.psi[1,:,:]).dot(H))
#                         print("forth part",self.gamma[1].dot(H))
                
                #reward2 = np.apply_along_axis(self.get_reward,1,states_next)
                reward[k,j] = reward1[k] + self.get_reward(states_next[j,:])#reward2[j]
#             if k == 32:
#                 print(states[k])
#                 print(states_next[1])
        q_value = np.sort(np.amax(reward,axis=1))
        print("top 5 q values",np.round(q_value[-5:],2))
#         print("last 5 q values",np.round(q_value[:5],2))
#         print("reward1",np.round(reward1[np.argsort(np.amax(reward,axis=1))[-5:]],2))
#         print("difference is",np.round(np.sort(np.amax(reward,axis=1))[-1]-np.sort(np.amax(reward,axis=1))[-4],2))
        print("best action",np.argsort(np.amax(reward,axis=1))[-5:] + 1)
#         print("worst action",np.argsort(np.amax(reward,axis=1))[:5] + 1)
        best_action = np.unravel_index(reward.argmax(), reward.shape)
        best_action = best_action[0] + 1
        #best_action = np.unravel_index(np.argsort(reward.ravel())[-5:], reward.shape)#[0] + 1 # get best-5-action
#         best_action = np.argsort(np.amax(reward,axis=1))[-4:] + 1
        if (reward1==np.zeros(self.action_range.size)).all():
            best_action = np.nan
        return best_action, np.amax(reward,axis=1)#q_value#, states[:,0]#, reward, reward1
    
    def fun(self, state_pre, m):
        if m==0:
            tmp = ((.1*state_pre[m,:])*((state_pre[m,:]<20) & (state_pre[m,:]>-20)) + 
                2 * (state_pre[m,:]>=20) + 
                -2 * (state_pre[m,:]<=-20))
#             tmp = ((-.2*state_pre[m,:])*((state_pre[m,:]<10) & (state_pre[m,:]>-10)) + 
#                 -2 * (state_pre[m,:]>=10) + 
#                 2 * (state_pre[m,:]<=-10))
        else:
            tmp = ((.2*state_pre[m,:])*((state_pre[m,:]<10) & (state_pre[m,:]>-10)) + 
                2 * (state_pre[m,:]>=20) + 
                -2 * (state_pre[m,:]<=-20))
#             tmp = ((-.2*state_pre[m,:])*((state_pre[m,:]<10) & (state_pre[m,:]>-10)) + 
#                 -2 * (state_pre[m,:]>=10) + 
#                 2 * (state_pre[m,:]<=-10))

#         tmp = np.clip(tmp,-3,3)
#         tmp -= 2
        return tmp


def sample_offline_data(n, n_timestep, n_interact_past_state, n_action, embed_dim, eta, logger, binary_vector=False, cat_vector=False, no_representative_vector=False, small_action_space=False):
    kappa_0 = np.load("kappa.npy",allow_pickle=True)
    kappa = np.load("kappa.npy",allow_pickle=True)
    if eta == 0.2:
        kappa = np.load("kappa_eta0.2.npy",allow_pickle=True)
    if eta == 0.8:
        kappa = np.load("kappa_eta0.8.npy",allow_pickle=True)
    index_regimens = np.load("index_regimens.npy",allow_pickle=True)
    phi = np.load("phi.npy",allow_pickle=True)
    action_space_name = np.load("action_space_name.npy",allow_pickle=True)
    drug_names = np.load("drug_names.npy")
    
    action_space_name_binary = np.zeros((action_space_name.shape[0],drug_names.shape[0]))
    for i in range(action_space_name.shape[0]):
        for j in range(drug_names.shape[0]):
            if sum(action_space_name[i,:] == drug_names[j]) == 1:
                action_space_name_binary[i,j] = 1

    drug_index_all, drug_times = np.unique(index_regimens[~np.isnan(index_regimens)], return_counts=True)

    n_action = n_action
    index_kernel_ori = np.sort(drug_index_all[np.argsort(-drug_times)[:n_action]])
    index_kernel_idx = index_kernel_ori.astype(int) - 1
    action_range = np.sort(drug_index_all[np.argsort(-drug_times)[:(1*n_action)]])
    if small_action_space:
        index_kernel_ori = np.array([7,8,9,19,20,28,51,57,71,72,73,111,118,135,141,144,145,147,153,159])
        action_range = index_kernel_ori

    
    #similarity_kappa = kappa[index_kernel_idx,:][:,index_kernel_idx]
    similarity_kappa = kappa[action_range.astype(int)-1,:][:,index_kernel_idx]
    similarity_kappa = similarity_kappa/similarity_kappa.sum(axis=1)[:,None]

    similarity_kappa = similarity_kappa - np.mean(similarity_kappa,axis=0)
    cov = np.cov(similarity_kappa,rowvar=False)
    evals, evecs = LA.eigh(cov)
    embed_dim = embed_dim
    sim_proj_all1 = similarity_kappa.dot(evecs[:,-embed_dim:])
    sim_proj_0 = sim_proj_all1
    #sim_proj_0 = similarity_kappa

    #np.save("index_kernel_0.npy",index_kernel_ori)
    #np.save("action_range_0.npy",action_range)
    np.save(os.path.join(logger.writer.get_logdir(), "index_kernel_0.npy"), index_kernel_ori)
    np.save(os.path.join(logger.writer.get_logdir(), "action_range_0.npy"), action_range)

    if binary_vector:
        action_space_name_binary = np.zeros((action_space_name.shape[0],drug_names.shape[0]))
        for i in range(action_space_name.shape[0]):
            for j in range(drug_names.shape[0]):
                if sum(action_space_name[i,:] == drug_names[j]) == 1:
                    action_space_name_binary[i,j] = 1

        #sim_proj_0 = action_space_name_binary[index_kernel_ori.astype(int)-1,:]
        sim_proj_0 = action_space_name_binary[action_range.astype(int)-1,:]
        
    if cat_vector:
        ## simply represent action as categorical
        sim_proj_0 = np.eye(len(action_range))
        
    if no_representative_vector:
        similarity_kappa = kappa[action_range.astype(int)-1,:][:,action_range.astype(int)-1]

        similarity_kappa = similarity_kappa - np.mean(similarity_kappa,axis=0)
        cov = np.cov(similarity_kappa,rowvar=False)
        evals, evecs = LA.eigh(cov)
        embed_dim = embed_dim
        sim_proj_all1 = similarity_kappa.dot(evecs[:,-embed_dim:])
        sim_proj_0 = sim_proj_all1

    #np.save("sim_proj_0.npy",sim_proj_0)
    np.save(os.path.join(logger.writer.get_logdir(), "sim_proj_0.npy"), sim_proj_0)
    
    # In[332]:
    np.random.seed(1)
    n = n
    J = n_timestep*np.ones(n).astype(int) + np.random.randint(5,size=n).astype(int)
    #np.save("J_0.npy",J)
    np.save(os.path.join(logger.writer.get_logdir(), "J_0.npy"), J)


    # In[401]:


    # time
    T = np.empty((n,max(J)))
    T[:] = np.nan
    for i in range(n):
        T[i,:J[i]] = np.arange(J[i]) * 0.5
    #np.save("T_0.npy",T)
    np.save(os.path.join(logger.writer.get_logdir(), "T_0.npy"), T)


    # In[335]:


    X = np.empty((n,max(J),3))
    X[:] = np.nan
    for i in range(n):
        X[i,:J[i],0] = 1
        X[i,:J[i],2] = 0
        X[i,:J[i],2] = np.random.randint(2,size=1)
        tmp = np.random.uniform(0,1)
        if tmp < 0.4:
            X[i,:J[i],1] = np.random.normal(2,0.1,1)
        elif tmp < 0.5:
            X[i,:J[i],1] = np.random.normal(-2,0.1,1)
        elif tmp < 0.65:
            X[i,:J[i],1] = np.random.normal(-1,0.1,1)
        else:
            X[i,:J[i],1] = np.random.normal(1,0.1,1)
        
    #np.save("X_0.npy",X)
    np.save(os.path.join(logger.writer.get_logdir(), "X_0.npy"), X)


    # In[336]:


    index_regimens_0 = np.empty((n,max(J)))
    index_regimens_0[:] = np.nan
    np.random.seed(1)
    #pp = np.ones(n_action)/n_action
    #pp = np.ones(action_range.size)/action_range.size
    
    pp = np.ones(n_action)/n_action
    #pp[:40] = 0.001
    #pp[40:] = (1-0.04)/(n_action-40)

    for i in range(n):
        for j in range(J[i]):
            index_regimens_0[i,j] = np.random.choice(action_range,size=1,p = pp)
            #index_regimens_0[i,j] = np.random.choice(index_kernel_ori,size=1,p = pp)
    #np.save("index_regimens_0.npy",index_regimens_0)
    
    #for i in range(n):
        #index_regimens_0[i,:J[i]] = action_range[(i % n_action)]
   #     index_regimens_0[i,5:J[i]] = action_range[((i+1) % n_action)]

    np.save(os.path.join(logger.writer.get_logdir(), "index_regimens_0.npy"), index_regimens_0)


    # In[337]:


    drug_history_0 = np.zeros((n,max(J),sim_proj_0.shape[1]))
    for i in range(n):
        for j in range(J[i]):
            drug_history_0[i,j,:] = sim_proj_0[np.where(action_range==index_regimens_0[i,j]),:]
            #drug_history_0[i,j,:] = sim_proj_0[np.where(index_kernel_ori==index_regimens_0[i,j]),:]

            
    if binary_vector:
        drug_history_0 = np.zeros((n,max(J),action_space_name_binary.shape[1]))
        for i in range(n):
            for j in range(J[i]):
                drug_history_0[i,j,:] = action_space_name_binary[index_regimens_0[i,j].astype(int) - 1,:]
                
    if cat_vector:
        drug_history_0 = np.zeros((n,max(J),len(action_range)))
        for i in range(n):
            for j in range(J[i]):
                drug_history_0[i,j,:] = (index_regimens_0[i,j] == action_range)

    #np.save("drug_history_0.npy",drug_history_0)
    np.save(os.path.join(logger.writer.get_logdir(), "drug_history_0.npy"), drug_history_0)


    
    # In[396]:


    


    # In[397]:


    y_ori = np.empty((n,4,max(J)))
    y_ori[:] = np.nan
    best_action = np.zeros(n)
    q_value = np.zeros((n,n_action))
    env = hivEnv(X,index_kernel_ori,action_range,index_regimens_0,kappa_0,n_interact_past_state,action_space_name_binary)
    logger.print("=============== sample offline data ==============")
    for i in range(n):
        #logger.print("patient {}".format(i))
        env.reset(i)
        for t in range(J[i]):
            if t == J[i]-3:
                best_action[i,], q_value[i,:] = env.get_opt_action()
            y_ori[i,:,t] = env.step(index_regimens_0[i,t].astype(int), T[i,t])[0]

    # In[400]:


    #np.save("y_0.npy",y_ori)
    #np.save("best_action.npy",best_action)
    np.save(os.path.join(logger.writer.get_logdir(), "best_action.npy"), best_action)
    np.save(os.path.join(logger.writer.get_logdir(), "y_0.npy"), y_ori)
    np.save(os.path.join(logger.writer.get_logdir(), "q_value.npy"), q_value)

    # In[154]:


#     def get_reward(state):
#         reward = (-state[0]-state[1]+state[2])/3
#         return reward


    # In[289]:


    ##### reformulate offline data
#     data = []
#     n = len(J)
#     for i in range(n):
#         current_state = np.concatenate([np.transpose(y_ori[i,:,:J[i]-2]), X[i,:J[i]-2,:]],axis=1)
#         next_state = np.concatenate([np.transpose(y_ori[i,:,1:J[i]-1]), X[i,:J[i]-2,:]],axis=1)
#         #action = drug_history_0[i,1:J[i]-1,:]
#         action = index_regimens_0[i,1:J[i]-1].reshape(-1,1).astype(int)
#         for j in range(J[i]-2):
#             action[j,:] = np.where(index_regimens_0[i,j]==index_kernel_ori)
#         reward = np.apply_along_axis(get_reward,1,current_state)
#         done = np.zeros(J[i]-2)
#         done[-1] = 1
#         tmp = {
#             "observations": current_state,
#             "next_observations": next_state,
#             "actions": action,
#             "rewards": reward,
#             "terminals": done
#         }
#         data.append(tmp)

#     offline_data = {}
#     for d in data:
#         for key, value in d.items():
#             offline_data.setdefault(key, []).append(value)
#     for key, value in offline_data.items():
#         offline_data[key] = np.concatenate(offline_data[key], axis=0)



    # In[265]:


    # this is trajectory-typed data

    # data = []
    # n = len(J)
    # act_dim = drug_history_0.shape[2]
    # for i in range(n):
    #     current_state = np.concatenate([np.transpose(y_ori[i,:,:J[i]-2]), X[i,:J[i]-2,:]],axis=1).reshape(1,-1,6)
    #     next_state = np.concatenate([np.transpose(y_ori[i,:,1:J[i]-1]), X[i,:J[i]-2,:]],axis=1).reshape(1,-1,6)
    #     action = drug_history_0[i,1:J[i]-1,:].reshape(1,-1,act_dim)
    #     reward = np.apply_along_axis(get_reward,2,current_state)
    #     done = np.zeros((1,J[i]-2))
    #     done[-1] = 1
    #     tmp = {
    #         "observations": current_state,
    #         "next_observations": next_state,
    #         "actions": action,
    #         "rewards": reward,
    #         "terminals": done
    #     }
    #     data.append(tmp)

    # offline_data = {}
    # for d in data:
    #     for key, value in d.items():
    #         offline_data.setdefault(key, []).append(value)
    # for key, value in offline_data.items():
    #     offline_data[key] = np.concatenate(offline_data[key], axis=0)


    # In[292]:


#     import pickle
#     with open('mopo/offline_simulation.pickle', 'wb') as handle:
#         pickle.dump(offline_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



