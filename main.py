import time
import numpy as np



class MountainCar:
    def __init__(self):
        self.state = [-0.5,0]
        self.actions = [-1,1]
        self.stateActionPairSize = len(self.state) + len(self.actions)

        self.m = -123123
    def startEpisode(self):
        self.state = [-0.5,0]

    def isEpisodeOver(self):
        return self.state[0] >= 0.6

    def getState(self):
        return np.float32(self.state)
        
    def act(self, a):
        a = self.actions[a] * 2
        self.state[1] = self.state[1] + 0.01 * a + np.cos(3*self.state[0])*-0.025
        self.state[0] = self.state[0] + self.state[1]
        if self.state[0] > self.m: self.m = self.state[0]
        if self.state[0] > 0.6:  self.state[0] = 0.6
        if self.state[0] < -1.2: self.state[0] = -1.2; self.state[1]=0;
        if self.state[1] > 0.07: self.state[1] = 0.07
        if self.state[1] <-0.07: self.state[1] =-0.07
        if self.state[0] >= 0.5: return 1
        return -1
    
class StupidCar:
    def __init__(self):
        self.state = [-0.5,0]
        self.actions = [-1,1]
        self.stateActionPairSize = len(self.state) + len(self.actions)

    def startEpisode(self):
        self.state = [-0.5,0]

    def isEpisodeOver(self):
        return self.state[0] >= 0.5

    def getState(self):
        return np.float32(self.state)
        
    def act(self, a):
        a = self.actions[a]
        self.state[1] = self.state[1] + 0.01 * a
        self.state[0] = self.state[0] + self.state[1]
        if self.state[0] > self.m: self.m = self.state[0]
        if self.state[0] > 0.6:  self.state[0] = 0.6
        if self.state[0] < -1.2: self.state[0] = -1.2; self.state[1]=0; 
        if self.state[1] > 0.07: self.state[1] = 0.07
        if self.state[1] <-0.07: self.state[1] =-0.07
        if self.state[0] >= 0.5: return 1
        return -1


# stolen from:
# https://gist.github.com/EderSantana/c7222daa328f0e885093
class Catch(object):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False
        

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return reward

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]

    # compatibility with my code:
    def isEpisodeOver(self):
        return self._is_over()
    def getState(self):
        return self.observe().flatten()
        #return self.observe().flatten()
    def startEpisode(self):
        self.reset()

import scipy.ndimage.filters as fl
import scipy.misc


class ALEGame:
    def __init__(self, ale):
        self.ale = ale
        self.actions = ale.getMinimalActionSet()
    def getState(self):
        def g(x):
            return fl.gaussian_filter(x, 2.5)
        x = self.ale.getScreenGrayscale()[:,:,0]
        if 1:
            x = np.float32(scipy.misc.imresize(x, (20,20)) / 255.)
            #x = np.float32(scipy.misc.imresize(x, (84,84)) / 255.)
        if 1:
            return g(x).flatten()
        return x[:,:,0]
    def isEpisodeOver(self):
        return self.ale.game_over()
    def act(self, action):
        return self.ale.act(action)
        #return sum(self.ale.act(action) for _ in range(4))
    def startEpisode(self):
        self.ale.reset_game()
            

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
        return kernel(s,sbar) / kernel(s,Sbar).sum(axis=-1)

    def update_Mbar():
        """ update pseudo MDP \bar{M}"""
        for a in range(nactions):
            Sa = S[a][0]
            Ra = S[a][1]
            Sah = np.asarray(S[a][2])
            n_a = len(Sa)
            zb = np.zeros(n_a)
            if not n_a: continue
            zb = kernel(Sah[:,None,:], Sbar[None,:,:]).sum(axis=-1)
            for i in range(m):
                zprime = (kernel(Sbar[i], Sa)).sum()
                for j in range(m):
                    b = (kernel(Sbar[i],Sa) * kernel(Sah, Sbar[j]) / zb).sum()
                    Pb[a,i,j] = 1./(z[a,i] + zprime) * (b+Pb[a,i,j] * z[a,i])
                
                e = (kernel(Sbar[i], Sa) * Ra).sum()
                rb[a,i] = 1./(z[a,i] + zprime) * (e + rb[a,i] * z[a,i])
                z[a,i] = z[a,i] + zprime

    def update_Qbar():
        """ perform value iteration to get the new \bar{Q}"""
        Q0 = Qbar
        for i in range(500):
            Q = (Pb.transpose(1, 0, 2) * (rb.T + gamma * Q0.max(axis=1)[:,None])[:,:,None]).sum(axis=2)
            if np.mean(abs(Q-Q0)) < 0.001: # converged
                break
            Q0 = Q
        Qbar[:] *= 0
        Qbar[:] += Q[:]
        
    epsilon = [.5]
    nticks = [0]
    last_t0 = [time.time()]
    def tick():
        s = mdp.getState()
        if np.random.random() < epsilon[0]:
            a = np.random.randint(0,nactions)
        else:
            qsa = np.float32([(kappa(s, Sbar) * Qbar[:,a]).sum()
                              for a in range(nactions)])
            a = np.random.choice(np.arange(nactions)[np.max(qsa) == qsa])
            
        
        r = mdp.act(a)
        if r > 0:
            epsilon[0] *= 0.95
            
        S[a][0].append(s)
        S[a][1].append(r)
        S[a][2].append(mdp.getState())
        nticks[0] += 1
        if not nticks[0] % tm:
            # make new random representative states
            if repr_strategy == 'random':
                for i in range(m):
                    Sa = []
                    while not len(Sa): Sa = S[np.random.randint(0,nactions)][0]
                    Sbar[i] = Sa[np.random.randint(0,len(Sa))]
            if repr_strategy == 'random_rewards':
                ls = 0
                tol = 0 
                while ls < m / 2 and tol < 1000:
                    tol += 1
                    idx = np.random.randint(0,nactions)
                    Sa,Ra,Sah = S[idx]
                    Sa = [s for s,r in zip(Sa,Ra) if r>0]
                    if not len(Sa): continue
                    Sbar[ls] = Sa[np.random.randint(0,len(Sa))]
                    ls += 1
                while ls < m:
                    idx = np.random.randint(0,nactions)
                    Sa,Ra,Sah = S[idx]
                    if not len(Sa): continue
                    Sbar[ls] = Sa[np.random.randint(0,len(Sa))]
                    ls += 1
                    
            # update Mbar and za
            update_Mbar()
            # reset Sa
            for a in range(nactions):
                S[a] = [[],[],[]]
                
        if not nticks[0] % tv:
            update_Qbar()
            print 'last ',tv,'ticks took',time.time()-last_t0[0],epsilon[0]
            last_t0[0] = time.time()
        return r
    return tick

from scipy.stats import norm

def gaussian(a,b,scale=0.2):
    return np.sum(norm.pdf(a-b, scale=scale),axis=-1)

max_steps = None

if 0:
    mdp = MountainCar()
    sigma = 0.02
    repr_strategy = 'random'
    tick = ikbsf(mdp, 50, 2, lambda a,b:gaussian(a,b,sigma), mdp.getState().shape[0], 200, 200)
    max_steps = 10000
    
if 0:
    mdp = StupidCar()
    sigma = 0.02
    repr_strategy = 'random'
    tick = ikbsf(mdp, 50, 2, lambda a,b:gaussian(a,b,sigma), mdp.getState().shape[0], 200, 200)
    max_steps = 10000

if 0:
    mdp = Catch(40)
    sigma = 0.5
    repr_strategy = 'random_rewards'
    tick = ikbsf(mdp, 50, 2, lambda a,b:gaussian(a,b,sigma), mdp.getState().shape[0], 200, 200)

if 1:
    import sys
    sys.path.append('/data/bengioe/atari/Arcade-Learning-Environment')
    import ale_python_interface
    ale = ale_python_interface.ALEInterface()
    ale.loadROM('/data/bengioe/atari/aleroms/breakout.bin')
    mdp = ALEGame(ale)
    nactions = len(mdp.actions)
    sigma = 0.5
    repr_strategy = 'random_rewards'
    tick = ikbsf(mdp, 50, nactions, lambda a,b:gaussian(a,b,sigma), mdp.getState().shape[0], 1000, 1000)
    
for i in range(1000):
    nsteps = 0
    R = 0
    mdp.startEpisode()
    while not mdp.isEpisodeOver():
        r = tick()
        R += r
        nsteps += 1
        if max_steps and nsteps > max_steps:
            break
    print 'episode',i,R,nsteps
