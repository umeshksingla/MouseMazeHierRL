import numpy as np
from MM_Traj_Utils import SplitModeClips, StepType, StepType2, StepType3


def ModeMask(tf,ma,re):
    '''
    Creates an array that parallels the bouts in tf giving the behavioral mode for every state
    0=home,1=drink,2=explore
    '''
    cl=SplitModeClips(tf,ma,re=re) # split the trajectory into mode clips
    ex=[np.zeros(len(b)) for b in tf.no] # list of arrays, one for each bout
    for c in cl: # for each clip
        ex[c[0]][c[1]:c[1]+c[2]]=c[3] # mark the states in the clip with the mode
    return ex


def FourBiasFit5(da, mk, ma):
    '''
    Computes 4 bias parameters from the node sequence da in maze ma
    da=sequence of node states
    mk=mask array, only consider actions into a state with mk==1
    Bf = prob of going forward when arriving from bottom of the T
    Ba = prob of making an alternating (instead of repeating) turn when going forward from bottom
    Lf = prob of going forward when arriving from L branch (or R branch)
    Lo = prob of turning out (instead of in) when going forward from L (or R)

    '''

    def Ratio(x, y):  # check for divide by zero
        if y == 0:
            return 0
        else:
            return x / y

    lo = 1;
    hi = 2 ** ma.le - 2  # [lo,hi]=range of node numbers for T-junctions excluding node 0
    tr = np.zeros((4,
                   4))  # transitions between steps: in left = 0; in right = 1; out left = 2; out right = 3
    st = np.array([-1] + [StepType(da[j - 1], da[j], ma) for j in
                          range(1, len(da))])  # step type that led to node i
    for i in range(1, len(st) - 1):  # correlate step i with step i+1
        if mk[i + 1] == 1:
            tr[st[i], st[i + 1]] += 1
    Bf = Ratio(tr[0, 0] + tr[0, 1] + tr[1, 0] + tr[1, 1],
               np.sum(tr[0, :]) + np.sum(
                   tr[1, :]))  # forward bias from bottom of T
    Ba = Ratio(tr[0, 1] + tr[1, 0], tr[0, 0] + tr[0, 1] + tr[1, 0] + tr[
        1, 1])  # alternating bias from bottom of T
    Lf = Ratio(tr[2, 1] + tr[2, 2] + tr[2, 3] + tr[3, 0] + tr[3, 2] + tr[3, 3],
               np.sum(tr[2, :]) + np.sum(
                   tr[3, :]))  # forward bias from the side
    Lo = Ratio(tr[2, 2] + tr[2, 3] + tr[3, 2] + tr[3, 3],
               tr[2, 1] + tr[2, 2] + tr[2, 3] + tr[3, 0] + tr[3, 2] + tr[
                   3, 3])  # out bias from the side
    return [Bf, Ba, Lf, Lo]


def FourBiasPredict5(be, da, ma):
    '''
    be=list of 4 biases
    da=sequence of node states
    ma=maze
    Given the bias array be and the sequence of node states da,
    computes the probability of each possible action at each place in the sequence
    '''

    def Ratio(x, y):  # check for divide by zero
        if y == 0:
            return 0
        else:
            return x / y

    Bf, Ba, Lf, Lo = be  # the four biases
    pro = np.array(
        # probabilities for the next action, depending on the 4 kinds of last step
        [[Bf * (1 - Ba), Bf * Ba, 1 - Bf],
         # 'in left' to ['in left','in right','out']
         [Bf * Ba, Bf * (1 - Ba), 1 - Bf],
         # 'in right' to ['in left','in right','out']
         [1 - Lf, Lf * (1 - Lo), Lf * Lo],
         # 'out left' to ['in left','in right','out']
         [Lf * (1 - Lo), 1 - Lf,
          Lf * Lo]])  # 'out right' to ['in left','in right','out']
    lo = 1;
    hi = 2 ** ma.le - 2  # [lo,hi]=range of node numbers for T-junctions excluding node 0
    st = np.array([-1] + [StepType(da[j - 1], da[j], ma) for j in
                          range(1, len(da))])  # step type that led to state i
    # step types are: in left = 0; in right = 1; out left = 2; out right = 3
    pr = np.zeros((len(da),
                   3))  # probabilities for the 3 kinds of action in left = 0; in right = 1; out = 2
    for i in range(1, len(st) - 1):  # predict action i+1 from step i
        if da[i] == 0:  # if we are at the first junction, node 0
            pr[i + 1] = [0.5, 0.5,
                         0]  # can only go left or right with equal prob
        elif da[i] <= hi:  # if this is a T-junction other than node 0
            pr[i + 1] = pro[
                st[i]]  # set the probabilities according to the 4 biases bias
        else:  # if this is an end node
            pr[i + 1] = [0, 0, 1]  # can only reverse
    return pr


def FourBiasWalkFit(tf,ma,tju=True,exp=True,rew=True,seg=5):
    '''
    Performs a Four-bias walk fit to predict actions in the trajectory tf on maze ma.
    tju = actions at T-junctions only?
    exp = actions in "explore" mode only?
    rew = animal rewarded (relevant only if exp==True)
    seg = number of segments for testing
    returns average cross-entropy per action
    '''
    if exp:
        ex=ModeMask(tf,ma,rew) # one array for each bout marking states with mode = 0,1,2
    ce=0
    for i in range(seg): # for each testing segment
        dte=np.concatenate([b[:-2,0] for b in tf.no[i::seg]]) # test states, 1/seg
        mte=np.ones(len(dte));mte[0:2]=0 # test mask, can't predict first 2 actions
        ate=np.array([-1]+[StepType2(dte[j-1],dte[j],ma) for j in range(1,len(dte))]) # actions to be predicted
        dtr=np.concatenate([b[:-2,0] for j in range(seg) for b in tf.no[j::seg] if j!=i]) # train states, 1-1/seg
        mtr=np.ones(len(dtr));mtr[0:2]=0 # train mask, can't predict first 2 actions
        if tju: # restrict to actions taken from a T-junction
            mtr[np.where(np.logical_or(dtr[:-1]<1,dtr[:-1]>62))[0]+1]=0
            mte[np.where(np.logical_or(dte[:-1]<1,dte[:-1]>62))[0]+1]=0
        if exp: # restrict to actions taken to an explore state
            exe=np.concatenate([e[:-2] for e in ex[i::seg]]) # test mode
            exr=np.concatenate([e[:-2] for j in range(seg) for e in ex[j::seg] if j!=i]) # train mode
            mte[np.where(exe!=2)[0]]=0 # limit the test mask to explore mode
            mtr[np.where(exr!=2)[0]]=0 # limit the train mask to explore mode
        be=FourBiasFit5(dtr,mtr,ma) # measure bias parameters from the training set
        pte=FourBiasPredict5(be,dte,ma) # use these to get probabilities for the testing set
        ce+=CrossEntropy5(pte[mte==1],ate[mte==1]) # evaluate the cross entropy
    ce/=seg
    return ce # average cross-entropy per action


def CrossEntropy5(p,s):
    '''
    evaluate cross entropy for multinomial distribution p[m,n] and data s[n]
    p = array of predicted probability
    s = data as index into p[]
    '''
    return -np.sum([np.log(p[i,s[i]]) for i in range(len(s))])/len(s)/np.log(2) # cross entropy per sample


# Markov models
# These Markov chains look at the history of the animal's preceding states leading up to the current state (node). This history is used as a pointer into a lookup table to output the probability of the 3 possible actions.
#
# The fixed-tree and variable-tree versions only differ in how they construct that lookup table.


# Action history
# When asking how the mouse got to a node, there are 3 possible preceding actions:
#
# in: This will be either left or right, but for any given node there was only one such option
# out right: stepping out of the maze along an R branch
# out left: stepping out of the maze along an L branch So we want a function that will assign those to integers 0,1,2 which we can use as indices.
# We have various routines that return the type of step between two nodes, for different purposes:
#
# StepType() makes in left = 0; in right = 1; out left = 2; out right = 3; illegal = -1
# StepType2() makes in left = 0; in right = 1; out = 2; illegal = -1
# StepType3() makes in = 0; out left = 1; out right = 2; illegal = -1


"""
FixMarkovTrain2(da,mk,de,ma)
This does the tallying of history-action combinations for fixed-length history of depth  𝑘= de. Returns the table of counts. This can then be turned into a probability estimate or analyzed otherwise.

The key here is to convert every possible history into an index into an array that can be used for accumulating counts. Here the history string is formulated as

(𝑎−(𝑘+2),…,𝑎−1,𝑎0,𝑠0 )

where  𝑠0  is the current state,  𝑎0  the action that led to that state, and another  𝑘−2  actions preceding that. Note this fully identifies the sequence of states, given the constraints from the maze.

There are 127 possible states  𝑠0  and 3 possible actions for each of the  𝑎𝑖 . So the resulting array will have  127∗3(𝑘−1)  components.

In this routine, the tuple  (𝑎−(𝑘+2),…,𝑎−1,𝑎0,𝑠0 ) is used directly for the indices of the counts array.

The mask array of Booleans mk marks which of the actions should be included in the fit. Note the current state immediately precedes the action in the array da. So if one wants to include only actions from nodes that are T-junctions, one needs to mark the element in mk after the occurrence of a T-junction node in da.
"""
def FixMarkovTrain2(da,mk,de,ma):
    '''
    Fixed depth Markov chain, training
    No restriction on nodes
    Collects counts for all combinations of history-action
    da = data
    mk = mask applied to action, Boolean
    de = depth of history
    ma = maze
    '''
    nt=2**(ma.le+1)-1 # number of possible nodes, rightmost symbol in history is current node
    sh=(3,)*(de-1)+(nt,)+(3,) # shape of counts array
    co=np.zeros(sh) # counts of history-action combinations
    ac=np.array([-1]+[StepType2(da[j-1],da[j],ma) for j in range(1,len(da))]) # forward actions
    hi=np.array([-1]+[StepType3(da[j-1],da[j],ma) for j in range(1,len(da))]) # history actions
    for i in range(de,len(da)): # i points to the action to be predicted
        if mk[i]:
            x=tuple(hi[i-de+1:i]) # start array pointer with history of preceding de-1 reverse actions
            x+=(da[i-1],) # add the most recent state
            co[x][ac[i]]+=1 # increase count for that action-history combination
    return co


"""
FixMarkovTest2(da,mk,pr,ma)
This does the evaluation of a fixed-depth Markov model on test data da. The model is contained in the array of probabilities pr that holds conditional probability

𝑝(𝑎1|𝐡) 
𝑎1  is the action following the current state  𝑠0  and

𝐡=(𝐚−(𝐤+2),…,𝐚−1,𝐚0,𝐬0)  is a history of length  𝑘  as described in FixMarkovTrain2().

Again you can select which actions should be evaluated with the mask array mk.
For every position in da it converts the preceding history into an array index, and uses that to look up the predicted probabilities in pr. The compares to the observed action and reports the cross entropy per symbol.
"""

def FixMarkovTest2(da,mk,pr,ma):
    '''
    Fixed depth Markov chain, testing
    da = data
    mk = mask applied to action, Boolean
    pr = probability array
    ma = maze
    '''
    sh=pr.shape # shape of probability array
    de=len(sh)-1 # depth of history
    ac=np.array([-1]+[StepType2(da[j-1],da[j],ma) for j in range(1,len(da))]) # forward actions
    hi=np.array([-1]+[StepType3(da[j-1],da[j],ma) for j in range(1,len(da))]) # history actions
    pt=[] # predicted probabilities for the observed action
    for i in range(de,len(da)): # i points to the action to be predicted
        if mk[i]:
            x=tuple(hi[i-de+1:i]) # start array pointer with history of preceding de-1 reverse actions
            x+=(da[i-1],) # add the most recent state
            pt+=[pr[x][ac[i]]] # add probability for the observed action to the list
    ce=-np.sum(np.log(pt))/len(pt)/np.log(2) # cross-entropy
    return ce


"""VarMarkovTrain2(da,mk,de,ma)
This makes a fit with a variable-depth history.

It starts out by tallying fixed-depth histories with a large depth  𝑘=6 . For this it simply uses FixMarkovTrain2().

Then it prunes those histories depending on often each happens in the training data. mc is the desired minimal count, there can be no history with fewer occurrences.

The full-length histories that have <mc counts simply inherit the history-action counts from the closest ancestor in the tree that has >=mc counts. We still leave the full array in place where every history has  𝑘=6  elements, but many of those will have the same 3-array of counts associated with the 3 actions.

The routine also returns a histogram of the length of the histories that meet the minimum count criterion.

Note the resulting array of counts can be used in the exact same way as that from FixMarkovTrain2().
"""
def VarMarkovTrain2(da,mk,mc,ma):
    '''
    Variable depth Markov chain, training
    Starts by collecting counts for a fixed depth of histories
    Then prunes those histories, such that every remaining branch has at least mc counts
    Does that by "fusing" twigs that are below the cutoff, i.e. making the counts equal on all those twigs
    da = data
    mk = mask applied to action, Boolean
    mc = minimum count for each history
    ma = maze
    returns:
    co=counts for each history-action combination
    br=histogram of history length, from l=1 to de
    '''
    de=6 # maximum depth allowed
    co=FixMarkovTrain2(da,mk,de,ma) # get counts for each history-action, use routine for fixed depth
    br=np.zeros(de)
    if de==1: # If depth==1 we're done already, no pruning of histories
        br[0]=len(co) # number of histories with length 1
    else: # If depth>1 start pruning of histories
        cou=[] # list of arrays holding cumulative counts of all lower levels
        cou+=[co] # history length de, the raw counts, this will get modified in the pruning process
        for k in range(1,de): # after this we have pro[k] = total counts for histories of length de-k, k=0,...,de-1
            cou+=[np.sum(cou[k-1],axis=0)] # sum over the earliest history element from lower level, length de-k
        for k in range(de-1,0,-1): # now go down the tree starting at level de-1 to see which children survive
            lo=np.where(np.sum(cou[k-1],axis=-1)<mc) # children whose sum of counts over actions is too small
            lp=set([tuple(x)[1:] for x in np.array(lo).T]) # set of locations of their parents
            br[de-k-1]=len(lp) # br[l]=number of branches that got cut at length l+1
            for p in lp: # for each of those parents
                for x in [0,1,2]: # for each of its children
                    cou[k-1][(x,)+p]=cou[k][p] # set counts equal to those of the parent
        br[de-1]=np.prod(co.shape[:-1]) # number of tips of the tree
        for l in range(de-1,0,-1):
            br[l]=br[l]-3*br[l-1] # number of branches of length l+1
    return co,br


"""MarkovFit2(tf,ma,var,tju,exp,rew,seg,train)
This performs Markov model training and testing on data in trajectory tf. 
Returns the cross-entropy and some fit parameters.

var=True gives a variable depth fit, var=False a fixed depth.

For fixed-depth fits, the whole procedure is repeated for depths 1,...,6. All the resulting cross-entropies are returned along with the list of depths.

For variable-depth fits, the whole procedure is repeated for minimum counts of mc= 2,3,5,10,20,40,80,160,320,640. For each of those fits, the average history length is recorded. Then all the cross-entropies are returned along with the history lengths.


tju=True restricts the fit and evaluation to actions from T-junctions.

exp=True restricts the fit and evaluation to actions into an 'exploration' state.
In that case set rew=True if the animal is rewarded.

seg sets the number of segments by which the data are divided for training and testing. 
The default is 5 segments, which means an 80/20 split of training/testing data. 
For a test set, it joins together every 5th bout throughout the experiment. 
The others are joined together for the training set. That gets repeated 5 different ways. 
Similarly for other values of seg.

train=True will evaluate the fit on the training set instead of the test set. 
Useful to look for overfitting, but otherwise should be set False.

Note the conversion of counts to conditional probabilities: Here I used "additive smoothing with a pseudo-count of 1". 
For at least one animal where I varied that, 1 was better than 1/2 or 2, gave lower entropies.
"""
def MarkovFit2(tf,ma,var=True,tju=True,exp=True,rew=True,seg=5,train=False):
    '''
    Performs a Markov chain fit to predict actions in the trajectory tf on maze ma.
    Fit is restricted to T-junctions not including 0
    var = variable history depth?
    tju = actions at T-junctions only?
    exp = actions in "explore" mode only?
    rew = animal rewarded (relevant only if exp==True)
    seg = number of segments for testing
    train = evaluate on the training data?
    '''
    if exp:
        ex=ModeMask(tf,ma,rew) # one array for each bout marking states with mode = 0,1,2
    if var:
        par=np.array([2,3,5,10,20,40,80,160,320,640]) # min counts for the variable history
    else:
        par=np.array([1,2,3,4,5,6]) # fixed depths
    hi=np.zeros(len(par)) # to store the avg history string length
    ce=np.zeros(len(par)) # to store the cross entropy
    for i in range(seg): # for each of the testing segments
        dte=np.concatenate([b[:-2,0] for b in tf.no[i::seg]]) # test states
        mte=np.array([False]+[True,]*(len(dte)-1)) # all actions OK except first
        dtr=np.concatenate([b[:-2,0] for j in range(seg) for b in tf.no[j::seg] if j!=i]) # train states
        mtr=np.array([False]+[True,]*(len(dtr)-1)) # all actions OK except first
        if tju: # restrict to actions taken from a T-junction, incl node 0
            mtr[np.where(dtr[:-1]>62)[0]+1]=False
            mte[np.where(dte[:-1]>62)[0]+1]=False
        if exp: # restrict to actions taken to an explore state
            exe=np.concatenate([e[:-2] for e in ex[i::seg]]) # test mode
            exr=np.concatenate([e[:-2] for j in range(seg) for e in ex[j::seg] if j!=i]) # train mode
            mte[np.where(exe!=2)[0]]=False # limit the test mask to explore mode
            mtr[np.where(exr!=2)[0]]=False # limit the train mask to explore mode
        for j,k in enumerate(par):
            if var:
                co,le=VarMarkovTrain2(dtr,mtr,k,ma)
                hi[j]+=np.sum(le*np.arange(len(le)))/np.sum(le)+1 # mean history length
            else:
                co=FixMarkovTrain2(dtr,mtr,k,ma)
                hi[j]+=k # history length
            su=np.sum(co,axis=-1).reshape(co.shape[:-1]+(1,)) # sum of counts over actions for each history
#             pc=2 # pseudocount
#             pr=(co+pc)/(su+3*pc) # smoothed estimator of probability
            pr=(co+1)/(su+3) # smoothed estimator of probability, pseudocount of 1
            if train:
                ce[j]+=[FixMarkovTest2(dtr,mtr,pr,ma)] # evaluate on same as training data
            else:
                ce[j]+=[FixMarkovTest2(dte,mte,pr,ma)] # evaluate on testing data
    hi/=seg
    ce/=seg
    si=np.argsort(hi) # sort by history length
    return hi[si],ce[si]


# POOLING
"""Pooling states to increase history counts Another attempt to increase the counts for each history involved pooling counts over multiple T-junctions in the maze that are closely related by symmetry. For example, all the T-junctions at the same level of the binary tree look locally similar, in that they all have corridors of identical length leading from the junction. One could try treating them all as belonging to the same history.

Various versions of this:

Pool all T-junctions in the entire maze. 
Pool all nodes in the same level Pool separately the L and the R nodes on same level.
How to implement it:

Keep the action string in the history, but change only the last symbol that 
identifies the current state. Two different nodes may now get the same state symbol. 
Provide the Train and Test routines with a translation array that converts node 
numbers to the pooled states FixMarkovTrain3(da,mk,de,ma,tr) This is 
like FixMarkovTrain2(da,mk,de,ma) with the added option of translating the node
sequence into states. This allows pooling of different locations in the maze 
to produce a reduced set of histories with larger counts.
"""
def FixMarkovTrain3(da,mk,de,ma,tr=None):
    '''
    Fixed depth Markov chain, training
    No restriction on nodes except as specified by mk
    Collects counts for all combinations of history-action
    da = data
    mk = mask applied to action, Boolean
    de = depth of history
    ma = maze
    tr = translation array for node numbers to states
    '''
    if tr is not None:
        nt=np.max(tr)+1 # number of values for states
        st=tr[da] # translated state numbers
    else:
        nt=2**(ma.le+1)-1 # number of possible nodes, rightmost symbol in history is current node
        st=da # no translation
    sh=(3,)*(de-1)+(nt,)+(3,) # shape of counts array
    co=np.zeros(sh) # counts of history-action combinations
    ac=np.array([-1]+[StepType2(da[j-1],da[j],ma) for j in range(1,len(da))]) # forward actions
    hi=np.array([-1]+[StepType3(da[j-1],da[j],ma) for j in range(1,len(da))]) # history actions
    for i in range(de,len(da)): # i points to the action to be predicted
        if mk[i]:
            x=tuple(hi[i-de+1:i]) # start array pointer with history of preceding de-1 reverse actions
            x+=(st[i-1],) # add the most recent state
            co[x][ac[i]]+=1 # increase count for that action-history combination
    return co


def VarMarkovTrain3(da,mk,mc,ma,tr):
    '''
    Variable depth Markov chain, training
    Starts by collecting counts for a fixed depth of histories
    Then prunes those histories, such that every remaining branch has at least mc counts
    Does that by "fusing" twigs that are below the cutoff, i.e. making the counts equal on all those twigs
    da = data
    mk = mask applied to action, Boolean
    mc = minimum count for each history
    ma = maze
    tr = translation array for node numbers to states
    returns:
    co=counts for each history-action combination
    br=histogram of history length, from l=1 to de
    '''
    de=6 # maximum depth allowed
    co=FixMarkovTrain3(da,mk,de,ma,tr) # get counts for each history-action, use routine for fixed depth
    br=np.zeros(de)
    if de==1: # If depth==1 we're done already, no pruning of histories
        br[0]=len(co) # number of histories with length 1
    else: # If depth>1 start pruning of histories
        cou=[] # list of arrays holding cumulative counts of all lower levels
        cou+=[co] # history length de, the raw counts, this will get modified in the pruning process
        for k in range(1,de): # after this we have pro[k] = total counts for histories of length de-k, k=0,...,de-1
            cou+=[np.sum(cou[k-1],axis=0)] # sum over the earliest history element from lower level, length de-k
        for k in range(de-1,0,-1): # now go down the tree starting at level de-1 to see which children survive
            lo=np.where(np.sum(cou[k-1],axis=-1)<mc) # children whose sum of counts over actions is too small
            lp=set([tuple(x)[1:] for x in np.array(lo).T]) # set of locations of their parents
            br[de-k-1]=len(lp) # br[l]=number of branches that got cut at length l+1
            for p in lp: # for each of those parents
                for x in [0,1,2]: # for each of its children
                    cou[k-1][(x,)+p]=cou[k][p] # set counts equal to those of the parent
        br[de-1]=np.prod(co.shape[:-1]) # number of tips of the tree
        for l in range(de-1,0,-1):
            br[l]=br[l]-3*br[l-1] # number of branches of length l+1
    return co,br


def FixMarkovTest3(da,mk,pr,ma,tr=None):
    '''
    Fixed depth Markov chain, testing
    da = data
    mk = mask applied to action, Boolean
    pr = probability array
    ma = maze
    tr = translation array for node numbers to states
    '''
    if tr is not None:
        st=tr[da] # translated state numbers
    else:
        st=da # no translation
    sh=pr.shape # shape of probability array
    de=len(sh)-1 # depth of history
    ac=np.array([-1]+[StepType2(da[j-1],da[j],ma) for j in range(1,len(da))]) # forward actions
    hi=np.array([-1]+[StepType3(da[j-1],da[j],ma) for j in range(1,len(da))]) # history actions
    pt=[] # predicted probabilities for the observed action
    for i in range(de,len(da)): # i points to the action to be predicted
        if mk[i]:
            x=tuple(hi[i-de+1:i]) # start array pointer with history of preceding de-1 reverse actions
            x+=(st[i-1],) # add the most recent state
            pt+=[pr[x][ac[i]]] # add probability for the observed action to the list
    ce=-np.sum(np.log(pt))/len(pt)/np.log(2) # cross-entropy
    return ce


def GetPR2(da,mk,pr,ma,tr=None):
    '''
    Fixed depth Markov chain, testing
    da = data
    mk = mask applied to action, Boolean
    pr = probability array
    ma = maze
    tr = translation array for node numbers to states
    '''
    if tr is not None:
        st=tr[da] # translated state numbers
    else:
        st=da # no translation
    sh=pr.shape # shape of probability array
    de=len(sh)-1 # depth of history
    ac=np.array([-1]+[StepType2(da[j-1],da[j],ma) for j in range(1,len(da))]) # forward actions
    hi=np.array([-1]+[StepType3(da[j-1],da[j],ma) for j in range(1,len(da))]) # history actions
    pt=[] # predicted probabilities for the observed action
    pt_dict = {}
    for i in range(de,len(da)): # i points to the action to be predicted
        if mk[i]:
            x=tuple(hi[i-de+1:i]) # start array pointer with history of preceding de-1 reverse actions
            x+=(st[i-1],) # add the most recent state
            prob = pr[x][ac[i]]
            pt+=[prob] # add probability for the observed action to the list
            pt_dict[x + (0,)] = pr[x][0]
            pt_dict[x + (1,)] = pr[x][1]
            pt_dict[x + (2,)] = pr[x][2]
    return pt, pt_dict


def MarkovFit3(tf,ma,var=True,tju=True,exp=True,rew=True,seg=5,train=False,transl=None):
    '''
    Performs a Markov chain fit to predict actions in the trajectory tf on maze ma.
    Fit is restricted to T-junctions not including 0
    var = variable history depth?
    tju = actions at T-junctions only?
    exp = actions in "explore" mode only?
    rew = animal rewarded (relevant only if exp==True)
    seg = number of segments for testing
    train = evaluate on training set?
    transl = array to convert node numbers into a reduced set of state numbers
    '''
    if exp:
        ex=ModeMask(tf,ma,rew) # one array for each bout marking states with mode = 0,1,2
    if var:
        par=np.array([2,3,5,10,20,40,80,160,320,640]) # min counts for the variable history
    else:
        par=np.array([1,2,3,4,5,6]) # fixed depths
    hi=np.zeros(len(par)) # to store the avg history string length
    ce=np.zeros(len(par)) # to store the cross entropy
    for i in range(seg):
        dte=np.concatenate([b[:-2,0] for b in tf.no[i::seg]]) # test states
        mte=np.array([False]+[True,]*(len(dte)-1)) # mask for testing, all actions OK except first
        dtr=np.concatenate([b[:-2,0] for j in range(seg) for b in tf.no[j::seg] if j!=i]) # train states
        mtr=np.array([False]+[True,]*(len(dtr)-1)) # mask for training, all actions OK except first
        if tju: # restrict to actions taken from a T-junction, incl node 0
            mtr[np.where(dtr[:-1]>62)[0]+1]=False # mask for training, eliminate end nodes
            mte[np.where(dte[:-1]>62)[0]+1]=False # mask for testing, eliminate end nodes
        if exp: # restrict to actions taken to an explore state
            exe=np.concatenate([e[:-2] for e in ex[i::seg]]) # test mode
            exr=np.concatenate([e[:-2] for j in range(seg) for e in ex[j::seg] if j!=i]) # train mode
            mte[np.where(exe!=2)[0]]=False # limit the test mask to explore mode
            mtr[np.where(exr!=2)[0]]=False # limit the train mask to explore mode
        for j,k in enumerate(par):
            if var:
                co,le=VarMarkovTrain3(dtr,mtr,k,ma,transl)
                hi[j]+=np.sum(le*np.arange(len(le)))/np.sum(le)+1 # mean history length
            else:
                co=FixMarkovTrain3(dtr,mtr,k,ma,transl)
                hi[j]+=k # history length
            su=np.sum(co,axis=-1).reshape(co.shape[:-1]+(1,)) # sum of counts over actions for each history
            pr=(co+1)/(su+3) # smoothed estimator of probability (note at node 0 there are just 2 options)
            if train:
                ce[j]+=FixMarkovTest3(dtr,mtr,pr,ma,transl) # evaluate on same as training data
            else:
                ce[j]+=FixMarkovTest3(dte,mte,pr,ma,transl) # evaluate on testing data
    hi/=seg
    ce/=seg
    si=np.argsort(hi) # sort by history length
    return hi[si],ce[si]


def MarkovFitAndTest3(tf_tr,tf_te,ma,var=True,tju=True,exp=True,rew=True,transl=None):
    '''
    Performs a Markov chain fit to predict actions in the trajectory tf_tr on maze ma.
    Returns cross entropy on the test trajectory tf_te
    Fit is restricted to T-junctions not including 0
    var = variable history depth?
    tju = actions at T-junctions only?
    exp = actions in "explore" mode only?
    rew = animal rewarded (relevant only if exp==True)
    train = evaluate on training set?
    transl = array to convert node numbers into a reduced set of state numbers
    '''
    if exp:
        ex_tr=ModeMask(tf_tr,ma,rew) # one array for each bout marking states with mode = 0,1,2
        ex_te=ModeMask(tf_te,ma,rew)  # one array for each bout marking states with mode = 0,1,2
    if var:
        par=np.array([2,3,5,10,20,40,80,160,320,640]) # min counts for the variable history
    else:
        par=np.array([1,2,3,4,5,6]) # fixed depths
    hi=np.zeros(len(par)) # to store the avg history string length
    ce=np.zeros(len(par)) # to store the cross entropy

    dte=np.concatenate([b[:-2,0] for b in tf_te.no]) # test states
    mte=np.array([False]+[True,]*(len(dte)-1)) # mask for testing, all actions OK except first
    dtr=np.concatenate([b[:-2,0] for b in tf_tr.no]) # train states
    mtr=np.array([False]+[True,]*(len(dtr)-1)) # mask for training, all actions OK except first
    if tju: # restrict to actions taken from a T-junction, incl node 0
        mtr[np.where(dtr[:-1]>62)[0]+1]=False # mask for training, eliminate end nodes
        mte[np.where(dte[:-1]>62)[0]+1]=False # mask for testing, eliminate end nodes
    if exp: # restrict to actions taken to an explore state
        exe=np.concatenate([e[:-2] for e in ex_te]) # test mode
        exr=np.concatenate([e[:-2] for e in ex_tr]) # train mode
        mte[np.where(exe!=2)[0]]=False # limit the test mask to explore mode
        mtr[np.where(exr!=2)[0]]=False # limit the train mask to explore mode

    for j,k in enumerate(par):
        if var:
            co,le=VarMarkovTrain3(dtr,mtr,k,ma,transl)
            hi[j]+=np.sum(le*np.arange(len(le)))/np.sum(le)+1 # mean history length
        else:
            co=FixMarkovTrain3(dtr,mtr,k,ma,transl)
            hi[j]+=k # history length
        su=np.sum(co,axis=-1).reshape(co.shape[:-1]+(1,)) # sum of counts over actions for each history
        pr=(co+1)/(su+3) # smoothed estimator of probability (note at node 0 there are just 2 options)
        ce[j]+=FixMarkovTest3(dte,mte,pr,ma,transl) # evaluate on testing data
    si=np.argsort(hi) # sort by history length

    return hi[si],ce[si]


def GetPR(tf,ma,var=True,tju=True,exp=True,rew=True,transl=None,par=2):
    '''
    Performs a Markov chain fit to predict actions in the trajectory tf on maze ma.
    Fit is restricted to T-junctions not including 0
    var = variable history depth?
    tju = actions at T-junctions only?
    exp = actions in "explore" mode only?
    rew = animal rewarded (relevant only if exp==True)
    seg = number of segments for testing
    transl = array to convert node numbers into a reduced set of state numbers
    par = min counts for the variable history OR fixed depth
    '''
    # print("exp, tju, transl, par", exp, tju, transl, par)
    if exp:
        ex=ModeMask(tf,ma,rew) # one array for each bout marking states with mode = 0,1,2
    if var:
        par=np.array(par) # min counts for the variable history
    else:
        par=np.array(par) # fixed depths

    dtr=np.concatenate([b[:-2,0] for b in tf.no]) # train states
    mtr=np.array([False]+[True,]*(len(dtr)-1)) # mask for training, all actions OK except first
    if tju: # restrict to actions taken from a T-junction, incl node 0
        mtr[np.where(dtr[:-1]>62)[0]+1]=False # mask for training, eliminate end nodes
        # print("masking endnodes")
    if exp: # restrict to actions taken to an explore state
        exr=np.concatenate([e[:-2] for e in ex]) # train mode
        mtr[np.where(exr!=2)[0]]=False # limit the train mask to explore mode
    k = par
    if var:
        co,le=VarMarkovTrain3(dtr,mtr,k,ma,transl)
        hi=np.sum(le * np.arange(len(le))) / np.sum(le) + 1  # mean history length
    else:
        co=FixMarkovTrain3(dtr,mtr,k,ma,transl)
        hi=k
    su=np.sum(co,axis=-1).reshape(co.shape[:-1]+(1,)) # sum of counts over actions for each history
    pr=(co+1)/(su+3) # smoothed estimator of probability (note at node 0 there are just 2 options)
    pt, pt_dict = GetPR2(dtr, mtr, pr, ma, transl)
    return hi, pr, pt, pt_dict


# make an array to translate nodes to states in groups {0},{t-junctions},{end nodes}
def TranslTJs(ma):
    '''
    discriminate by type of node; treat entry junction as separate from other T-junctions.
    '''
    tra=np.zeros(2**(ma.le+1)-1,dtype=int) # number of node values
    tra[0]=0 # entry junction
    tra[1:2**(ma.le)-1]=1 # other t junctions
    tra[2**(ma.le)-1:]=2 # end nodes
    return tra


def TranslLevels(ma):
    '''
    discriminate by level in the maze
    '''
    tra=np.zeros(2**(ma.le+1)-1,dtype=int) # number of node values
    for k in range(ma.le+1): # nodes level k
        tra[2**k-1:2**(k+1)-1]=k
    return tra


def TranslLevelsLR(ma):
    '''
    discriminate by level and whether an L or an R node
    '''
    tra=np.zeros(2**(ma.le+1)-1,dtype=int) # number of node values
    tra[0]=0 # nodes level 0
    for k in range(1,ma.le+1): # nodes level 1,...
        for j in range(2**k-1,2**(k+1)-1):
            tra[j]=2*k-1+StepType(ma.pa[j],j,ma) # This distinguishes L from R nodes
    return tra
