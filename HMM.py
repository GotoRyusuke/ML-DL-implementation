# -*- coding: utf-8 -*-
'''
Implementations of Hidden Markov Models

'''
import numpy as np

class HMM:
    def __init__(self):
        pass
    
    @staticmethod
    def drawing_chain():
        pass
    
    @staticmethod
    def foreward(obs_pre_t, start_probs, trans, emit):
        # attributes of the prob. matrix
        _, nstate = trans.shape
        nobs = len(obs_pre_t)
        
        # initialise the prob. matrix 
        alpha_mat = np.zeros([nstate, nobs])
        
        for s in range(nstate):
            alpha_mat[s, 0] = start_probs[s] * emit[obs_pre_t[0]-1, s]
        for t in range(1,nobs):
            for s in range(nstate):
                alpha_mat[s,t] = sum(alpha_mat[:,t-1]*trans[:,s]*emit[obs_pre_t[t]-1,s])    
        
        return alpha_mat
    
    @staticmethod
    def backward(obs_pos_t, start_betas, trans, emit):
        _, nstate = trans.shape
        nobs = len(obs_pos_t)
        
        # initialise beta matrix at time t
        beta_mat = np.zeros([nstate, nobs])
        beta_mat[:,-1] = 1
        
        for t in range(1,nobs):
            for i in range(nstate):
                beta_mat[i,nobs-t-1] = sum(beta_mat[:, nobs-t] * trans[i, :] * emit[obs_pos_t[nobs-t]-1, :])
                 
        return beta_mat
        
    
    @staticmethod
    def viterbi(obs,start_probs, trans, emit):
        # attributes of the prob. matrix
        _, num_states = trans.shape
        num_obs = len(obs)
        
        # initialise the prob. matrix 
        prob_mat = np.zeros([num_states, num_obs])
        back_mat = np.zeros([num_states, num_obs])
        
        for s in range(num_states):
            prob_mat[s, 0] = start_probs[s] * emit[obs[0]-1, s]
            back_mat[s,0] = 0
        for t in range(1, num_obs):
            for s in range(num_states):
                temp_probs = prob_mat[:,t-1]*trans[1:,s]*emit[obs[t]-1,s]
                prob_mat[s,t] = max(temp_probs)
                back_mat[s,t] = np.argmax(temp_probs)
        best_prob = max(prob_mat[:,-1])
        best_pointer = np.argmax(prob_mat[:,-1])
        best_path = np.zeros([num_obs], dtype = int)
        best_path[-1] = best_pointer
        
        for t in range(1,num_obs):
            node = num_obs - t
            from_ = back_mat[best_path[node], node]
            best_path[node-1] = from_
            
        return best_path, best_prob
    
    @staticmethod
    def BaumWelch(obs, start_probs,
                  init_trans, init_emit, start_betas,  
                  max_iter = 10e4, thres = 10e-5):
        _, nstate = init_trans.shape
        nobs = len(obs)
        niter = 0
        si_mat = np.zeros([nstate, nstate, nobs])# for any i,j,t
        
        while niter < max_iter:
            alpha_mat = HMM.foreward(obs, start_probs, init_trans, init_emit)
            beta_mat = HMM.backward(obs, start_betas, init_trans, init_emit)
            denom = sum(beta_mat[:,0] * init_emit[obs[0]-1,:] * start_probs)
            gamma_mat = alpha_mat * beta_mat/denom
            
            new_trans = np.zeros(init_trans.shape)
            new_emit = np.zeros(init_emit.shape)

            for i in range(nstate):
                for j in range(nstate):
                    for t in range(nobs-1):
                        si_mat[i,j,t] = alpha_mat[i,t] * init_trans[i,j] * init_emit[obs[t]-1, j] * beta_mat[j,t]/denom 
                    new_trans[i,j] = si_mat[i,j,:].sum()/si_mat[i,:,:].sum()
            for j in range(nstate):
                for k in range(len(init_emit)):
                    new_emit[k,j] = gamma_mat[j, obs == k].sum()/gamma_mat[j,:].sum()
            
            fwd_prob = HMM.foreward(obs, start_probs, init_trans, init_emit)
            new_fwd_prob = HMM.foreward(obs, start_probs, new_trans, new_emit)
            init_trans = new_trans
            init_emit = new_emit 
            niter += 1
            diff = abs(fwd_prob.sum() - new_fwd_prob.sum())
            print(diff)
            print(new_trans, '\n', new_emit)
            if  diff < thres:
                break
        return new_trans, new_emit
                
        # while niter < max_iter:
        
        

if __name__ == '__main__':
    
    trans = np.array([[0.6,0.4],
                      [0.5,0.5]])
    start_probs = [0.8,0.2] # [p(H start), p(C start)]
    emit = np.array([[.2,.4,.4],
                     [.5,.4,.1]]).T # 1-st col: p(x|H); 2-nd col: p(x|C)
    obs = np.array([3,1,3,2,3,1,3])
    state_dict = {'H':0, 'C':1}
    obs_dict = {'1':0, '2':1, '3':2}
    
    # fwd_test = HMM.foreward(obs,start_probs, trans, emit)
    # print('test foreward:\n', fwd_test)
    
    # start_betas= [1,1]
    # bwd_test = HMM.backward(obs, start_betas, trans, emit)
    # print('test bakforward:\n', bwd_test)
    # viTest_path, viTest_prob = HMM.viterbi(obs, start_probs, trans, emit)
    
    init_trans = np.array([[0.8, 0.2],
                           [0.2,0.8]])
    init_emit = np.array([[0.1,0.6],
                          [0.3,0.3],
                          [0.6,0.1]])
    start_betas = [1,1]
    
    test_trans, test_emit = HMM.BaumWelch(obs, start_probs, init_trans, init_emit, start_betas, max_iter=10)
    
    
