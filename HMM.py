# -*- coding: utf-8 -*-
'''
Implementations of Hidden Markov Models

'''
import numpy as np

class HMM:
    def __init__():
        pass
    
    @staticmethod
    def drawing_chain():
        pass
    
    @staticmethod
    def forward(obs, trans, emit):
        # attributes of the prob. matrix
        _, num_states = trans.shape
        num_obs = len(obs)
        
        # initialise the prob. matrix 
        prob_mat = np.zeros([num_states, num_obs])
        
        for s in range(num_states):
            prob_mat[s, 0] = trans[0, s] * emit[obs[0]-1, s]
        for t in range(1,num_obs):
            for s in range(num_states):
                temp_probs = prob_mat[:,t-1]*trans[1:,s]*emit[obs[t]-1,s]
                prob_mat[s,t] = temp_probs.sum()
        
        forward_prob = prob_mat[:,-1].sum()
        return round(forward_prob,5)
    
    @staticmethod
    def viterbi(obs,trans, emit):
        # attributes of the prob. matrix
        _, num_states = trans.shape
        num_obs = len(obs)
        
        # initialise the prob. matrix 
        prob_mat = np.zeros([num_states, num_obs])
        back_mat = np.zeros([num_states, num_obs])
        
        for s in range(num_states):
            prob_mat[s, 0] = trans[0, s] * emit[obs[0]-1, s]
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

if __name__ == '__main__':
    trans = np.array([[0.6,0.4],
                  [0.5,0.5]])
    pi = np.array([[0.8,0.2]]) # [p(H start), p(C start)]
    trans = np.concatenate([pi, trans])
    emit = np.array([[.2,.4,.4],
                     [.5,.4,.1]]).T # 1-st col: p(x|H); 2-nd col: p(x|C)
    obs = np.array([3,1,3])
    # forward_test = HMM.forward(obs,trans, emit)
    viTest_path, viTest_prob = HMM.viterbi(obs, trans, emit)
    print(viTest_path)
