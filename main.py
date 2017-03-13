import time
import numpy as np



class MountainCar:
    def __init__(self):
        self.state = [-0.5,0]
        self.actions = [-1,0,1]
        self.stateActionPairSize = len(self.state) + len(self.actions)

        self.m = -123123
    def startEpisode(self):
        self.state = [-0.5,0]

    def isEpisodeOver(self):
        return self.state[0] >= 0.6

    def getState(self):
        return np.float32(self.state)
        
    def act(self, a):
        a = self.actions[a]
        a = a + np.random.normal(0,0.1)
        self.state[1] = self.state[1] + 0.01 * a + np.cos(3*self.state[0])*-0.025
        self.state[0] = self.state[0] + self.state[1]
        #print self.state,self.m,"\r",
        if self.state[0] > self.m: self.m = self.state[0]
        if self.state[0] > 0.6:  self.state[0] = 0.6
        if self.state[0] < -1.2: self.state[0] = -1.2; self.state[1]=0; print "bump"
        if self.state[1] > 0.07: self.state[1] = 0.07
        if self.state[1] <-0.07: self.state[1] =-0.07
        if self.state[0] >= 0.5: return 1
        return -1#abs(self.state[1])*10
    
class StupidCar:
    def __init__(self):
        self.state = [-0.5,0]
        self.actions = [-1,0,1]
        self.stateActionPairSize = len(self.state) + len(self.actions)

        self.m = -123123
    def startEpisode(self):
        self.state = [-0.5,0]

    def isEpisodeOver(self):
        return self.state[0] >= 0.5

    def getState(self):
        return np.float32(self.state)
        
    def act(self, a):
        a = self.actions[a]
        #a = a + np.random.normal(0,0.1)
        self.state[1] = self.state[1] + 0.01 * a# + np.cos(3*self.state[0])*-0.025
        self.state[0] = self.state[0] + self.state[1]
        #print self.state,self.m,"\r",
        if self.state[0] > self.m: self.m = self.state[0]
        if self.state[0] > 0.6:  self.state[0] = 0.6
        if self.state[0] < -1.2: self.state[0] = -1.2; self.state[1]=0; 
        if self.state[1] > 0.07: self.state[1] = 0.07
        if self.state[1] <-0.07: self.state[1] =-0.07
        if self.state[0] >= 0.5: return 1
        return -1#abs(self.state[1])*10


def ikbsf(mdp,m,nactions,kernel,state_size,tm,tv):
    Pb = np.zeros((nactions, m, m))
    rb = np.zeros((nactions, m))
    z = np.zeros((nactions,m))
    Sbar = np.zeros((m, state_size))
    Qbar = np.zeros((m, nactions))
    # list of lists of (s,r,s')
    S = [[[],[],[]] for i in range(nactions)]
    gamma = 0.99
    
    def kappa(s,sbar):
        return kernel(s,sbar) / sum(kernel(s,sj) for sj in Sbar)

    def update_Mbar():
        for a in range(nactions):
            Sa = S[a][0]
            Ra = S[a][1]
            Sah = S[a][2]
            n_a = len(Sa)
            zb = np.zeros(n_a)
            for t,s_t in enumerate(Sa):
                zb[t] = sum(kernel(Sah[t], s_l) for s_l in Sbar)
            for i in range(m):
                zprime = sum(kernel(Sbar[i], Sa[t]) for t in range(n_a))
                for j in range(m):
                    b = sum(kernel(Sbar[i],Sa[t]) * kernel(Sah[t], Sbar[j]) / zb[t] for t in range(n_a))
                    Pb[a,i,j] = 1./(z[a,i] + zprime) * (b+Pb[a,i,j] * z[a,i])
                e = sum(kernel(Sbar[i], Sa[t]) * Ra[t] for t in range(n_a))
                rb[a,i] = 1./(z[a,i] + zprime) * (e + rb[a,i] * z[a,i])
                z[a,i] = z[a,i] + zprime

    def update_Qbar():
        # (m, na) = ((m, na, m) * (m, na,1) + (m,1,1)) -> sum axis=2
        #print Pb
        print rb
        #print Sbar
        Q0 = Qbar
        for i in range(500):
            Q = (Pb.transpose(1, 0, 2) * (rb.T + gamma * Q0.max(axis=1)[:,None])[:,:,None]).sum(axis=2)
            #Q = np.float32([[np.sum(Pb[a,s] * (rb[a] + gamma * Q0.max(axis=1)))
            #                 for a in range(nactions)]
            #                for s in range(m)])
            #print i,Q,Pb[0,0].sum()
            if np.mean(abs(Q-Q0)) < 0.001:
                break
            Q0 = Q
            #import pdb; pdb.set_trace()
        print Q
        print i, np.mean(abs(Q-Q0))
        Qbar[:] *= 0
        Qbar[:] += Q[:]
        

    nticks = [0]
    last_t0 = [time.time()]
    def tick():
        s = mdp.getState()
        if np.random.random() < 0.1:
            a = np.random.randint(0,nactions)
        else:
            qsa = np.float32([sum(kappa(s, Sbar[i]) * Qbar[i,a] for i in range(m))
                              for a in range(nactions)])
            a = np.random.choice(np.arange(nactions)[np.max(qsa) == qsa])
            
        
        r = mdp.act(a)
        #print nticks[0],s,a,r
        S[a][0].append(s)
        S[a][1].append(r)
        S[a][2].append(mdp.getState())
        nticks[0] += 1
        if not nticks[0] % tm:
            # make new random representative states
            for i in range(m):
                Sa = S[np.random.randint(0,nactions)][0]
                Sbar[i] = Sa[np.random.randint(0,len(Sa))]
            # update Mbar and za
            update_Mbar()
            # reset Sa
            for a in range(nactions):
                S[a] = [[],[],[]]
        if not nticks[0] % tv:
            update_Qbar()
            print 'last ',tv,'ticks took',time.time()-last_t0[0]
            last_t0[0] = time.time()
        return r
    return tick

from scipy.stats import multivariate_normal as mvnorm

def gaussian(a,b):
    #print a,b,mvnorm.pdf(a-b,mean=0,cov=.1)
    return np.mean(mvnorm.pdf(a-b,mean=0,cov=.1))
mdp = MountainCar()
mdp = StupidCar()
tick = ikbsf(mdp, 10, 3, gaussian, 2, 200, 200)

for i in range(100):
    nsteps = 0
    R = 0
    mdp.startEpisode()
    while not mdp.isEpisodeOver():
        r = tick()
        R += r
        nsteps += 1
        if nsteps > 1000:
            break
    print 'episode',i,R,nsteps
