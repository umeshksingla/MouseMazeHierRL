import numpy as np

from MM_Maze_Utils import StepType
from MM_Traj_Utils import TransMatrix
from parameters import EXPLORE

# A function to compute 6-parameter bias
def Bias2(i, m, trb, alt=False):
    '''
    Computes 6 biases for node i of maze m based on transition probs in trb
    Can enter the T junction from bottom (B), left (L), or right (R) of the T
    From each direction, can either go forward (f) or reverse back to the preceding state. (Bf, Lf and Rf will be used to
    describe the probability of going forward)
    If go forward from L or R, can go out (o) or into the maze. (Lo and Ro quantify the probability of going out)
    If go forward from B, can go left (l) or right. (Bl quantifies the probability of going left)
    Alternatively, if alt==True score this as an alternating (a) vs same direction turn.
    So the components of the bias are: Bf, Bl, Lf, Lo, Rf, Ro
    If a particular entry (B, L, or R) never occurs then returns -1 for those 2 bias values.
    '''

    def Norm(x):
        if x[0] == 0:
            return 0
        else:
            return x[0] / sum(x)

    tr = trb[i]  # 2-state transitions for this node

    B = np.sum(tr[0])
    if B > 0:
        Bf = (tr[0, 1] + tr[0, 2]) / B  # Bf = forward bias from B
        if Bf > 0:
            if alt and StepType(m.pa[i], i, m) == 0:  # an L node, so 'alt' refers to right turns
                Bl = tr[0, 2] / (tr[0, 2] + tr[0, 1])
            else:
                Bl = tr[0, 1] / (tr[0, 2] + tr[0, 1])  # Bl = left bias when stepping forward from B
        else:
            Bl = -1  # can't evaluate Bl bias
    else:
        Bf = -1;
        Bl = -1  # can't evaluate Bf and Bl bias

    L = np.sum(tr[1])
    if L > 0:
        Lf = (tr[1, 0] + tr[1, 2]) / L  # Lf = forward bias from L
        if Lf > 0:
            Lo = tr[1, 0] / (tr[1, 0] + tr[1, 2])  # Lo = outward bias if forward from L
        else:
            Lo = -1  # can't evaluate Lo bias
    else:
        Lf = -1;
        Lo = -1  # can't evaluate Lf and Lo bias

    R = np.sum(tr[2])
    if R > 0:
        Rf = (tr[2, 0] + tr[2, 1]) / R  # Rf = forward bias from R
        if Rf > 0:
            Ro = tr[2, 0] / (tr[2, 0] + tr[2, 1])  # Ro = outward bias if forward from R
        else:
            Ro = -1  # can't evaluate Ro bias
    else:
        Rf = -1;
        Ro = -1  # can't evaluate Rf and LRo bias
    return np.array([[Bf, Bl], [Lf, Lo], [Rf, Ro]])



# Derive 4 bias parameters from mouse trajectory, restricted to clips of type mode
# This version also returns the standard error across nodes
def ComputeFourBiasClips2(tr,ma,clips,mode=EXPLORE):
    '''
    Computes 4 bias parameters from trajectory `tr` in maze `ma`
    Limits itself to the clips of behavior type mode
    Bf = prob of going forward when arriving from bottom of the T
    Ba = prob of making an alternating (instead of repeating) turn when going forward from bottom
    Lf = prob of going forward when arriving from L branch (or R branch)
    Lo = prob of turning out (instead of in) when going forward from L (or R)
    :return list of 4 means, list of 4 stds (one element for each bias)
    '''
    sta,trb=SecondTransProbClips(tr,ma,clips,mode)
    # compute 6-parameter bias for each node without entry or end nodes
    bi=np.array([Bias2(i,ma,trb,alt=True) for i in range(1,2**ma.le-1)])
    Bf=-1;Ba=-1;Lf=-1;Lo=-1 # return -1 if no valid measurement
    Bfs=-1;Bas=-1;Lfs=-1;Los=-1 # return -1 if no valid measurement
    x=bi[:,0,0]
    i=x>-1
    if len(x[i])>0:
        Bf=np.mean(x[i])
        Bfs=np.std(x[i])
    x=bi[:,0,1]
    i=x>-1
    if len(x[i])>0:
        Ba=np.mean(x[i])
        Bas=np.std(x[i])
    x=np.concatenate((bi[:,1,0],bi[:,2,0]))
    i=x>-1
    if len(x[i])>0:
        Lf=np.mean(x[i])
        Lfs=np.std(x[i])
    x=np.concatenate((bi[:,1,1],bi[:,2,1]))
    i=x>-1
    if len(x[i])>0:
        Lo=np.mean(x[i])
        Los=np.std(x[i])
    return [Bf,Ba,Lf,Lo],[Bfs,Bas,Lfs,Los]


# Compute 2nd order transition probabilities within clips of one behavior mode
def SecondTransProbClips(tr,ma,clips,mode=EXPLORE):
    '''
    Computes 2nd order transition probabilities among nodes of maze ma in trajectory tr
    Limits itself to the clips of behavior type mode
    Returns arrays sta(n,3) and trb(n,3,3) with n=# of nodes
    sta[i,:] the 3 nodes connected to node i in order o,L,R; i.e. parent, left child, right child
    trb[i,j,k] is the prob of transition from state i to sta[i,k] given prior state was sta[i,j]
    '''
    sta=TransMatrix(ma) # array of nodes connected to each node, in order parent, left child, right child
    # make array of transition probabilities based on current and preceding state
    ta = TallyStringsClips(tr,clips,mode,3) # all occurrences of strings up to length 3
    trb=np.zeros((len(ma.ru),3,3)) # 3D array containing transition probability depending on 2-string
    for i in range(len(ma.ru)): # i is current state
        for j,sj in enumerate(sta[i]): # sta[i,j] is preceding state
            for k,sk in enumerate(sta[i]): # sta[i,k] is next state
                if (sj,i,sk) in ta[2]:
                    trb[i,j,k]=ta[2][(sj,i,sk)]/ta[1][(sj,i)]
    return sta,trb

# Tally occurrences for each j-string within clips of one behavior mode
def TallyStringsClips(tr,clips,mode=EXPLORE,m=5):
    '''
    Produces m dictionaries that give the number of occurrences for each j-string
    up to j=m in the trajectory tr.
    The strings of different length are aligned to all share the first element.
    Limits itself to the clips of behavior type mode
    Returns: a list of dictionaries of type s[string tuple]=number.
    '''
    se=[{} for i in range(m)] # se[j] tallies (j+1)-strings
    nt=tr.no
    for cl in clips:
        if cl[3]==mode:
            b=nt[cl[0]][cl[1]:cl[1]+cl[2]+1] # the piece of bout of this clip, incl first and last node
            for i in range(0,len(b)-m+1): # first position of the string
                for j in range(m): # j+1 = length of string
                    s=tuple(b[i:i+j+1,0]) # j+1-string starting at i
                    if s in se[j]: # tally into the j-th dictionary
                        se[j][s]+=1
                    else:
                        se[j][s]=1
    return se
