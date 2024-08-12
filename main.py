import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import math
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class mine_net(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = self.fc2(output)
        return output

class mine:
    def __init__(self, p_dis, q_dis, num_iterations, all = True, batch_size = 5000,lr = 0.001):
        self.lr = lr
        self.all = all
        self.ma_window_size = int(num_iterations/5)
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.obs = p_dis.round(decimals =2)
        self.acs = q_dis.round(decimals =2)
        #print(self.obs.shape, '1')
        if len(self.obs.shape) == 1:
            self.obs = np.expand_dims(self.obs, 1)
        if len(self.acs.shape) == 1:
            self.acs = np.expand_dims(self.acs, 1)
        #print(self.obs.shape, '2')
        self.expts =3

    def kullback_liebler(self, dis_p, dis_q, kl_net):
        t = kl_net(dis_p)
        et = torch.exp(kl_net(dis_q))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et)) #- (0.01*2**torch.log(torch.mean(et))*self.obs.shape[1])
        return mi_lb, t, et

    def learn_klne(self, batch, mine_net, mine_net_optim, ma_et, ma_rate=0.001):
        joint, marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
        mi_lb, t, et = self.kullback_liebler(joint, marginal, mine_net)
        ma_et = (1-ma_rate) * ma_et + ma_rate * torch.mean(et)
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach() * torch.mean(et))
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et
   
    def trip_sample_batch(self, sample_mode='joint'):
        index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
        #print(self.obs.shape, self.acs.shape, 'here')
        if sample_mode == 'marginal':
            marginal_index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
            
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[marginal_index, :])),axis=1)
        else:
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[ index, :])),axis=1)
        
        return batch

    def trip_train(self, tripmine_net, tripmine_net_optim):
        ma_et = 1.
        result = list()
        for i in range(self.num_iterations):
            batch = self.trip_sample_batch(), self.trip_sample_batch(sample_mode='marginal') 
            mi_lb, ma_et = self.learn_klne(batch, tripmine_net, tripmine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
        return result

    def ma(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]

    def trip_initialiser(self):
        tripmine_net = mine_net(self.obs.shape[1]+self.acs.shape[1]).to(device)
        tripmine_net_optim = optim.Adam(tripmine_net.parameters(), lr=self.lr)
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append(self.ma(self.trip_train( tripmine_net, tripmine_net_optim)))
        return np.array(trip_results)

    
    def run(self):
        results = self.trip_initialiser()
        return results
    
if __name__ == "__main__":
    N=10
    num_iters = 5
    X1 = np.random.choice([0, 1], size=N)
    X2 = np.random.choice([0, 1], size=N)
    X3 = np.random.choice([0, 1], size=N)
    Y = X1+X2+X3
    results = mine(np.vstack((X1, X2, X3)).T, Y, num_iterations=num_iters).run()
    print(f'average MI = {results.mean(axis=0)[-1]}, with std = {results.std(axis=0)[-1]}')

