#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from Sobol import i4_sobol_generate


# In[2]:


def STOCHOS(norm_power,
            norm_powerL,
            el_price,
            el_priceL,
            op_time,
            op_timeL,
            rle,
            access_LTH,
            theta=None,
            rho=None,
            curtailment=0,
            scalars=None,
            STH_sched=None,
            return_all_vars=False,
            rel_gap = 0.01,
            CMS = False,
            TBS = False
           ):
    '''
    STOCHOS -> Stochastic implementation of HOST
    
    Inputs:
    - norm_power: np.array of shape (STH_dim,N_t,N_S) -> hourly normalized power scenarios of the wind turbines in the STH
    - norm_powerL: np.array of shape (LTH_dim,N_t,S) -> daily normalized power scenarios of the wind turbines in the LTH
    - el_price: np.array of shape (STH_dim,N_S) -> hourly electricity price scenarios in the STH [$/MWh]
    - el_priceL: np.array of shape (LTH_dim,N_S) -> daily electricity price scenarios in the LTH [$/MWh]
    - op_time: np.array of shape (STH_dim,N_t,N_S) -> hourly operation time scenarios for turbine maintenance [hours]
    - op_timeL: np.array of shape (LTH_dim,N_t,N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - rle: np.array of shape (N_t,N_S) -> residual life estimate scenarios for each wind turbine [days]
    - access_LTH: np.array of shape (LTH_dim,N_t,N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - theta: (optional) np.array of shape (N_t,) -> binary parameter denoting whether the wind turbine requires maintenance.
        By default all turbines will require maintenance.. theta = np.ones(N_t)
    - rho: (optional) np.array of shape (N_t,) -> binary parameter denoting whether maintenance was initiated in the past.
        By default no maintenance was initiated in the past.. rho = np.zeros(N_t)
    - curtailment: (optional) np.array of shape (STH_dim,N_S) -> hourly power curtailment (normalized). Default value is zero. 
    - scalars: (optional) tuple or list containing scalar parameter values with the following order:
        Kappa (default: 4000) Cost of each PM task [$]
        Fi (default: 10000) Cost of each CM task [$]
        Psi (default: 250) Maintenance crew hourly cost [$/h]
        Omega (default: 2500) Vessel daily rental cost [$/d]
        Q (default: 125) Maintenance crew overtime cost [$/h]
        R (default: 12) Wind turbine rated power output [MW]
        W (default: 8) Number of workhours with standard payment [hours]
        B (default: 2) Number of maintenance crews
        H (default: 8) Max overtime hours in total [hours]
        tR (default: 5) Time of first sunlight
        tD (default: 21) Time of last sunlight
    - STH_sched: (optional) -> list containing pd.DataFrame with the fixed day-ahead PM schedule, if the user wants to simulate 
        the day ahead. In this case, the true parameter values should be used as deterministic datapoints (i.e. N_S=1). 
        Default value is None. 
    - return_all_vars: (optional) bool -> Returns the whole optimization model instead of selected processed outputs. 
        Default value is False to minimize memory usage.
    - rel_gap: (optional) float -> Sets the relative optimality gap of the optimizer.
        
    Returns:
    - output: A dictionary (only if return_all_vars is set to False) with the following keys: 
        'STH_PM_sched': pd.DataFrame of shape (STH_dim,N_t) with the PM schedule of the STH.
        'STH_CM_sched': pd.DataFrame of shape (STH_dim,N_t) with the CM schedule of the STH. 
        'LTH_PM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic PM schedule of the LTH. 
        'LTH_CM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic CM schedule of the LTH.
        'expected_STH_cost': The expected cost of the STH.
        'expected_LTH_cost': The expected cost of the LTH.
        'total_expected_cost': Total expected cost, which is the sum of the STH and LTH cost, plus the expected cost
            incured from unfinished maintenance tasks in the STH.
        'remaining_maint_hrs': np.array of shape (N_t, N_S) with the number of maintenance hours remaining for tasks that
            where initiated in the STH but not completed in the same day.
        'model': None, or a dict containing  all the variable values of the optimization model (only if 
            return_all_vars is set to True)
    '''
    
    ## Construct the sets ##
    tF, N_t, N_S = norm_power.shape
    dJ = norm_powerL.shape[0]
    
    I = ['wt'+str(i) for i in range(1,N_t+1)]
    T = ['t'+str(t) for t in range(tF)]
    D = ['d'+str(d) for d in range(1,dJ+1)]
    S = ['s'+str(s) for s in range(1,1+N_S)]
    
    
    ## Specify default parameters ##
    rho = pd.Series(np.zeros(N_t), index = I) if rho is None else pd.Series(rho, index=I)
    theta = pd.Series(np.ones(N_t), index = I) if theta is None else pd.Series(theta, index=I)
    
    C = pd.DataFrame(np.ones((tF,N_S))-curtailment, columns=S, index=T).T
    
    # Scalar parameters and costs
    if scalars is None:
        Kappa = 4000      #($/PM task) Cost of each PM task
        Fi = 10000        #($/CM task) Cost of each CM task
        Psi = 250         #($/h) Maintenance crew hourly cost
        Omega = 2500      #($/d) Vessel daily rental cost
        Q = 125           #($/h) Maintenance crew overtime cost
        R = 12            #(MW) Wind turbine rated power output
        W = 8             #(ts) Number of workts with standard payment
        B = 2             #(-) Number of maintenance crews
        H = 8
        tR = 5   
        tD = 21
    else:
        Kappa,Fi,Psi,Omega,Q,R,W,B,H,tR,tD = scalars
    
    Qa = 5000
    
    # Additional parameters
    rle = rle.copy()
    
    zeta = np.ones((N_t))
    zeta[np.all(rle==0, axis=1)] = 0
    zeta = pd.Series(zeta, index=I)

    zetaLong = np.ones((dJ,N_t,N_S))
    zetaLong[np.repeat(rle[np.newaxis,:,:],dJ,0)-np.arange(dJ).reshape((-1,1,1)) <= 1] = 0
    #print(rle[0,:])
    #print(zetaLong[:,0,:])
    zetaL = {d: {i: {s: zetaLong[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    
    ## Convert stochastic parameters to appropriate formats (dict) ##
    Pi = {t:{s: el_price[ti,si] for si, s in enumerate(S)} for ti, t in enumerate(T) }
    PiL = {d:{s: el_price[di,si] for si, s in enumerate(S)} for di, d in enumerate(D) }
    norm_power2 = norm_power.copy()
    norm_power2[norm_power2<=1e-4]=1e-4
    f = {t: {i: {s: norm_power2[ti,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for ti, t in enumerate(T)}
    fL = {d: {i: {s: norm_powerL[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    A = {t: {i: {s: op_time[ti,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for ti, t in enumerate(T)}
    AL = {d: {i: {s: op_timeL[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    
    #mission_time = repair_time + {frac of repair time that will be inaccessible}
    #mission_time = repair_time + repair_time*(1-access)
    A_LTH = {d: {i: {s: (1 + (1-access_LTH[di,ii,si])) for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}

    
    
    ## Initiate model instance and declare variables ##
    owf = gp.Model('STOCHOS')
    owf.update()

    # Continuous variables
    p = owf.addVars(T, I, S, lb=0, name="p") # hourly power generation for the STH
    pL = owf.addVars(D, I, S, lb=0, name="pL") # Daily power generation for the LTH
    l_STH = owf.addVar(lb=-np.inf, name = "l_STH") #Profit obtained in the STH
    l_LTH = owf.addVars(D, lb=-np.inf, name = "l_LTH") #Profit obtained in d d of the LTH

    # Integer variable
    q = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "q") # Overtime ts
    qL = owf.addVars(D, S, lb = 0, vtype = gp.GRB.INTEGER, name = "qL") # Overtime ts
    qa = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "qa") # Overtime ts
    xa = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "xa") # Overtime ts
    b = owf.addVars(I, S, lb = 0, vtype = gp.GRB.INTEGER, name = "b") # Overtime ts
    
    

    # Binary Variables
    m = owf.addVars(T, I, vtype = gp.GRB.BINARY, name = "m") 
    mL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "mL") 
    y = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "y")
    yL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "yL")
    v = owf.addVar(vtype = gp.GRB.BINARY, name = "v")
    vL = owf.addVars(D, S, vtype = gp.GRB.BINARY, name = "vL")
    x = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "x")
    w = owf.addVars(I, S, vtype = gp.GRB.BINARY, name = "w")
    z = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "z")
    
    ## Simulation mode ##
    if STH_sched is not(None):
        owf.reset(0)
        print("*** Initiating day-ahead simulation from input schedule ***")
        for i in I:
            for t in T:

                if STH_sched[i][t]==1:
                    m[t,i].lb=1
                else:
                    m[t,i].ub=0
                    
    else:
        print("#########################################################")
        print("# STOCHOS -> Stochastic Holistic Opportunistic Strategy #")
        print("#########################################################")


    #owf.update()



    U = {i: {s: Omega+R*PiL[D[0]][s]*fL[D[0]][i][s]*tR for s in S}
         for i in I} #Upfront costs from not finishing an initiated task
    Y = {i: {s: R*PiL[D[0]][s]*fL[D[0]][i][s] for s in S} 
         for i in I} #hourly costs from not finishing an initiated task

    
    # Constraints
    obj_fun = l_STH + gp.quicksum(l_LTH[d] for d in D) - 1/N_S*(gp.quicksum(
        U[i][s]*w[i,s]+Y[i][s]*b[i,s]*A_LTH['d1'][i][s] for i in I for s in S) + gp.quicksum(Qa*(xa[s]+qa[s]) for s in S))  #objective function

    con2 = owf.addConstr((l_STH == -gp.quicksum((1-rho[i])*(Kappa+(1-zeta[i])*(Fi-Kappa))*m[t,i]
                                               for t in T for i in I)-Omega*v+
                          1/N_S*gp.quicksum( gp.quicksum(Pi[t][s]*p[t,i,s]-Psi*x[t,i,s] 
                                                        for t in T for i in I)
                                            -Q*q[s] for s in S)), name = "STH profit")

    con3 = owf.addConstrs((l_LTH[d] == 1/N_S*
                           gp.quicksum(gp.quicksum(PiL[d][s]*pL[d,i,s]-(1.-rho[i])*(
                               Kappa+(1-zetaL[d][i][s])*(Fi-Kappa))*mL[d,i,s]-Psi*AL[d][i][s]*(
                               mL[d,i,s]) for i in I)-Omega*vL[d,s]-Q*qL[d,s] for s in S) for d in D), 
                          name = "LTH profit")

    con4 = owf.addConstrs((gp.quicksum(m[t,i] for t in T)+
                          gp.quicksum(mL[d,i,s] for d in D) == 
                          theta[i] for i in I for s in S), name = "Force maintenance")

    con5 = owf.addConstrs((m[t,i] <= (T.index(t))/tR for t in T for i in I), 
                          name = "Maintenance after sunrise")

    con6 = owf.addConstrs((m[t,i] <= tD/(1.01+T.index(t)) for t in T for i in I), 
                          name = "Maintenance before sunset")

    con7 = owf.addConstrs((gp.quicksum(z[T[T.index(t)+t_hat],i,s] 
                                       for t_hat in range(min([len(T[T.index(t):]),A[t][i][s]])))
                          >= min([len(T[T.index(t):]),A[t][i][s]])*m[t,i] 
                           for t in T for i in I for s in S
                          if T.index(t)+1.01 <= tD), name = "Downtime from ongoing maintenance")

    con8 = owf.addConstrs((b[i,s] >= gp.quicksum(m[t,i]*max([0,A[t][i][s]-len(T[T.index(t):])]) 
                                                    for t in T) for i in I for s in S), 
                          name = "Remaining hours of unfinished maintenance")

    con9 = owf.addConstrs((w[i,s] >= b[i,s]/100 for i in I for s in S), name = "Unfinished maintenance")

    con10 = owf.addConstrs((x[t,i,s] >= z[t,i,s]-(T.index(t))/tD for t in T for i in I for s in S), 
                           name = "Crew occupacy")

    con11 = owf.addConstrs((gp.quicksum(x[t,i,s] for i in I) <= B + xa[s]  for t in T for s in S), 
                           name = 'Max tasks per t')    #+ ba[s]<------------------------------

    con16 = owf.addConstrs((y[t,i,s] <= zeta[i]*(1-rho[i])
                           + gp.quicksum((len(T[T.index(tt):]))*m[tt,i] for tt in T)/
                             (len(T[T.index(t):])+0.1) + 1.0-theta[i]
                             for t in T for i in I for s in S), name = 'Availability STH')

    con17 = owf.addConstrs((yL[d,i,s] <= zetaL[d][i][s]*(1-rho[i])
                            +(dJ-gp.quicksum((D.index(dd)+1)*mL[dd,i,s] for dd in D))/
                            (dJ-D.index(d)-0.9) +(1.0-theta[i]) for d in D for i in I for s in S),
                            name = "Availability LTH")

    con18 = owf.addConstrs((y[t,i,s] <= 1 - z[t,i,s] for t in T for i in I for s in S),
                           name = "Unavailability from maintenance")

    con19 = owf.addConstr((v>=1/N_t*gp.quicksum(m[t,i] for t in T for i in I)), 
                           name = "STH vessel rental")

    con20 = owf.addConstrs((vL[d,s] >= 1/N_t*gp.quicksum(mL[d,i,s] for i in I) 
                            for d in D for s in S), name = "LTH vessel rental")

    con21 = owf.addConstrs((gp.quicksum(x[t,i,s] for t in T for i in I)<=B*W+q[s]+qa[s] for s in S), 
                          name = "Overtime")

    con22 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s]+b[i,s] for i in I) 
                            <= B*W+qL[d,s] for d in D for s in S if D.index(d)==0), 
                           name = 'Overtime 1st day of LTH')

    con23 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s] for i in I) 
                            <= B*W+qL[d,s] for d in D for s in S if D.index(d)>0), 
                           name = 'Overtime other days of LTH')

    con24 = owf.addConstrs((q[s]<=H for s in S), name = "STH max overtime")

    con25 = owf.addConstrs((qL[d,s]<=H for d in D for s in S), name = "LTH max overtime")

    con26 = owf.addConstrs((p[t,i,s]<=R*(f[t][i][s])*y[t,i,s] 
                            for t in T for i in I for s in S),
                           name = "STH power")

    con27 = owf.addConstrs((pL[d,i,s]<=24*R*(fL[d][i][s])*(yL[d,i,s]-mL[d,i,s]*zetaL[d][i][s]*AL[d][i][s]/24) 
                            for d in D for i in I for s in S),name = "LTH power")

    con28 = owf.addConstrs((gp.quicksum(p[t,i,s] for i in I) <= 
                           gp.quicksum(f[t][i][s] for i in I)*R*C[t][s] for t in T for s in S), 
                           name = 'Power curtailment')
    
    if CMS:
        cms_con = owf.addConstrs((gp.quicksum(m[t,i] for t in T) <= 1-zeta[i] for i in I), 
                           name = 'CMS benchmark constraint')
    
    if TBS:
        assert len(S)==1, 'TBS is a deterministic benchmark.'
        tbs_con = owf.addConstrs((gp.quicksum(m[t,i] for t in T) == theta[i]-zetaL['d1'][i]['s1'] for i in I), 
                           name = 'TBS benchmark constraint')
        

    #########################################################################################################


    # Set objective
    owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

    owf.setParam("MIPGap", rel_gap)
    owf.setParam("TimeLimit", 3600)

    owf.update()
    

    # Solve model
    owf.optimize()
    
    
    if owf.solCount == 0:
        return None
    else:

        STH_sched = pd.DataFrame(np.round(np.array(owf.getAttr('X',m).values()).reshape(-1,len(I))), 
                                 columns = I, index = T)

        LTH_sched = np.array(owf.getAttr('X',mL).values()).reshape(dJ,N_t,N_S)

        expected_STH_cost = (1/N_S*np.sum([f[t][i][s]*R*Pi[t][s]*C[t][s] for t in T for i in I for s in S])-l_STH.X)
        expected_LTH_cost = (1/N_S*np.sum([24*fL[d][i][s]*R*PiL[d][s] for d in D for i in I for s in S])-
                             np.sum(owf.getAttr('X',l_LTH).values()))
        total_expected_cost = expected_STH_cost + expected_LTH_cost + 1/N_S*np.sum([U[i][s]*owf.getAttr('X',w)[i,s]+
                                                                                  Y[i][s]*owf.getAttr('X',b)[i,s] 
                                                                                  for i in I for s in S])

        output = {'STH_sched': STH_sched, 
                  'LTH_sched': LTH_sched, 
                  'expected_STH_cost': expected_STH_cost,
                  'expected_LTH_cost': expected_LTH_cost,
                  'total_expected_cost': total_expected_cost,
                  'remaining_maint_hrs': np.array(owf.getAttr('X',b).values()).reshape(N_t,N_S),
                  'model': {var.varName: owf.getVarByName(var.varName).X for var in owf.getVars()} if return_all_vars else None
                 }
        return output


# In[3]:


def hourly2daily_aggregation(data):
    """
    Performs daily aggregation of hourly data.
    
    Input:
    - data: np.array of shape (hours, N_t, N_S)
    
    Returns:
    - output: np.array of shape (hours//24, N_t, N_S) with the daily mean of the input array
    """
    total_days = data.shape[0]//24
    output = np.array([np.mean(data[24*x:24*(x+1)],0) for x in range(total_days)])
    return output


# In[4]:

class RidgeRegression():
    
    def __init__(self, reg=0.001):
        """
        A Ridge Regression model with regulariation parameter defined by the user (default=0.001)
        """
        self.reg = reg
    
    def fit(self, X, y):
        """
        Returns the ridge regression coefficients that minimize the squared error of the data provided.
        
        Inputs:
        -X: np float array of shape (obs_dim, feat_size)
        -Y: np float array of shape (obs_dim, 1)
        
        Returns:
        -b: np floar array of shape (feat_size, 1)
        -train_MAE: float -> Mean Absolute Error of the training set
        """
        self.b = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.reg*np.eye(X.shape[-1])),np.matmul(X.T,y))
        train_MAE = np.abs(np.matmul(X,self.b)-y).mean()
        return self.b, train_MAE
    
    def predict(self, X):
        out = np.matmul(X,self.b)
        out[out<0]=0
        out[out>1]=1
        return out


#%%

def quadKernelAugmentedSpace(X):
    """
    Maps input space of shape (obs_dim, 2) to augmented space of shape (obs_dim, 6) of a quadratic Kernel.
    
    The first column of X is the daily avg wind speed and the second column is the daily avg wave height.
    """
    out = np.append(np.sqrt(2)*X,X**2,1)
    out = np.append(out, (np.sqrt(2)*X[:,0]*X[:,1]).reshape(-1,1),1)
    out = np.append(out, np.ones((X.shape[0],1)) ,1)
    return out

#%%

def get_mission_time_LTH(wind_speed_forecast_LTH, wave_height_forecast_LTH, tau, wave_lim=1.5, coefs=None):
    """
    Returns the expected hours of mission time for each day of the LTH, given the
    average daily wind speed and wave height
    
    Inputs:
    - wind_speed_forecast_LTH: np float array of shape (dJ, N_t, N_S)
    - wave_height_forecast_LTH: np float array of shape (dJ, N_S)
    - tau: np int array of shape (N_t,)
    - wave_lim: wave height safety threshold (1.5m, 1.8m or 2m) to use precomputed values of regression parameters.
    - coefs: np float arry of shape (1,6) -> coefficients of the regression using quadratic kernel space. 
        If the default value of None is used, one of the 3 precomputed values will be used for wave_lim.
    
    Returns:
    - mission_time: np float array of shape (dJ, N_t, N_S) -> LTH mission time scenarios
    - access_LTH: np float array of shape (dJ, N_t, N_S) -> daily accessibility fraction scenarios
    """
    (dJ, N_t, N_S) = wind_speed_forecast_LTH.shape
    #weights of augmented space from RidgeRegression(0.0001) with quadratic Kernel 
    #using historical NWP data and true accessibility data, to calculate the 
    #percentage of hours in a day of the LTH where the turbine will be inaccessible
    #Posted here to slightly reduce the complexity
    if coefs==None:
        if wave_lim==1.5:
            b = np.array(  [[ 0.02008826],
                            [0.76945388],
                            [-0.00409112],
                            [ 0.06934806],
                            [ 0.02252065],
                            [ 1.68621608]])  
        elif wave_lim==1.8:
            b = np.array(  [[ 0.03513277],
                            [-0.58401405],
                            [-0.00536439],
                            [ 0.02710852],
                            [ 0.02153183],
                            [ 1.47014811]])
        elif wave_lim==2:
            b = np.array(  [[ 4.18107897e-02],
                            [-4.62873380e-01],
                            [-6.23088680e-03],
                            [-7.37855258e-04],
                            [ 2.19760977e-02],
                            [ 1.32768890e+00]])
    else: b=coefs
    
    mission_time = np.zeros((dJ, N_t, N_S))
    access_LTH = np.zeros((dJ, N_t, N_S))
    for i in range(N_t):
        for s in range(N_S):
            X_in = np.append(wind_speed_forecast_LTH[:,i,s].reshape(-1,1),
                             wave_height_forecast_LTH[:,s].reshape(-1,1), 1)
            X = quadKernelAugmentedSpace(X_in)
            
            access_LTH[:,i,s] = np.matmul(X.copy(),b).reshape(-1) #(dJ,)
            access_LTH[access_LTH<0] = 0
            access_LTH[access_LTH>1] = 1
            #mission_time = repair_time + {frac of repair time that will be inaccessible}
            #mission_time = repair_time + repair_time*(1-access)
            mission_time[:,i,s] = tau[i]*(1 + (1-access_LTH[:,i,s]))
            
    
    return mission_time, access_LTH
            
#%%
def normal_gen_sobol(size):
    '''
    Generates a seqence of random numbers drawn from a standard
    normal distribution using the Sobol method to generate a uniform sample
    and the polar coordinates method as the tranformation technique.
    '''
    U = i4_sobol_generate(2, size)
    U1 = U[:,0]
    U2 = U[:,1]
    D = -2*np.log(U1)
    Theta = 2*np.pi*U2
    return np.sqrt(D)*np.cos(Theta)

def MVN_gen(mean_y, cov_y, N_S):
    '''
    Multi Variate Normal low discrepancy sequence generator
    '''
    pred_len = mean_y.shape[0]
    # Create L
    
    L = np.linalg.cholesky(cov_y)
    # Sample X from standard normal with Sobol
    X = normal_gen_sobol(size=pred_len*N_S).reshape(-1,N_S)
    # Apply the transformation
    Y = L.dot(X) + mean_y.reshape(pred_len,1)
    return Y

def error_scenario_gen(
        forecast_error_hist,
        pred_len,
        N_S,
        custom_hist_len = 24*10,
        random_seed=1,
        LDS = False,
        DGP=False
        ):

    """
    Scenario generation for deterministic forecast errors using 
    sklearn.gaussian_process.GaussianProcessRegressor fitting and sampling.
    
    Inputs:
    - forecast_error_hist: np float array of shape (hist_len,) -> Full error history
    - pred_len: int -> prediction length
    - N_S: int -> Number of scenarios to generate
    - *Optional*
        - custom_hist_len: int -> Custom history length. If hist_len<custom_hist_len,
            then hist_len is used (default=24*10)
        - random_seed=1
        
    Returns:
    -forecast_error_scenarios: np float array of shape (pred_len, N_S)
    """
    hist_len = forecast_error_hist.shape[0]
    
    forecast_error_scenarios = np.zeros((pred_len, N_S))
    
    if custom_hist_len>hist_len: custom_hist_len=hist_len
    
    kernel = RBF()
    gp_model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha = np.random.uniform(size=(custom_hist_len))**2,
            random_state=random_seed,  
            n_restarts_optimizer=10)
    
    X = np.arange(custom_hist_len).reshape(-1,1)
    Y = forecast_error_hist[hist_len-custom_hist_len:]
    gp_model.fit(X, Y)
    
    X_pred = (X[-1]+np.arange(1,pred_len+1)).reshape(-1,1)
    
    if np.all(Y==0):
        forecast_error_scenarios = np.zeros((pred_len, N_S))
    elif DGP:
        mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
        forecast_error_scenarios = mean_y.reshape(-1,1)
    else:
        if LDS:
            mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
            cov_y = cov_y+np.eye(cov_y.shape[0])*1e-6
            scenario_set = MVN_gen(mean_y, cov_y, 1000)
            sorted_std = np.argsort(np.std(scenario_set,0))
            forecast_error_scenarios = scenario_set[:,sorted_std[:N_S]]
        else:
            forecast_error_scenarios = gp_model.sample_y(X_pred,N_S,random_state=random_seed)
        
    return forecast_error_scenarios


# In[5]:


def el_price_scenario_gen_v1(el_prices_mean, 
                             el_price_std = 4,
                             N_S=100,
                             random_seed=1
                            ):
    
    """
    Scenario generation for electricity prices normal distribution.
    
    Inputs:
    - el_prices: np.array of floats of shape (obs_dim,)
    - el_price_std: float or np.array of floats of shape (obs_dim,1)
    """
    np.random.seed(random_seed)
    
    obs_dim = el_prices_mean.shape[0]

    el_prices = np.random.normal(loc = el_prices_mean.reshape(-1,1), scale = el_price_std, 
                                            size = (obs_dim,N_S))
    
    el_prices[el_prices<10] = 10
    
    return el_prices


# In[6]:


def rle_scenario_gen(
        rle_mean, 
        rle_std,
        N_S,
        dist = 'normal',
        random_seed=1
        ):

    """
    Scenario generation for RLE using normal distribution.
    
    Inputs:
    - rle_mean: np float array of shape (N_t,)
    - rle_std: float or np float array of shape (N_t,)
    -dist: str -> the distibution to be used for the scenario generation. Valid inputs are 'normal' and 'weibull'.
        If 'weibull' is selected, the rle_std is used as the shape parameter, and rle_mean as the scale.
    """
    np.random.seed(random_seed)
    N_t = rle_mean.shape[0]
    if not(hasattr(rle_std, '__len__')): rle_std = np.repeat(rle_std, N_t, 0)
    
    if dist.lower() == 'normal':
        rle = np.random.normal(loc = rle_mean.reshape(-1,1), scale = rle_std.reshape(-1,1), 
                                                size = (N_t,N_S))
    elif dist.lower() == 'weibull':
        rle = rle_mean.reshape(-1,1)*np.random.weibull(rle_std.reshape(-1,1), 
                                                size = (N_t,N_S))
        rle[rle_std==0, :] = rle_mean[rle_std==0].reshape(-1,1)
    else: 
        print("Valid RLE pdf's are 'normal' and 'weibull'.")
        
    rle[rle<1] = 1
    
    rle[np.repeat(rle_mean.reshape(-1,1)==0,N_S,-1)] = 0 #if the expected rle is 0 then we know with certainty that the 
                                                         # turbine has failed
    
    return rle


# In[7]:


def binning_method(wind_speed, bins=None):
    """
    Performs the binning method to calculate the normalized power from wind speed time series data.
    
    Inputs:
    - wind_speed: np.array of floats of shape (24*dJ, N_t, N_S) -> Wind speed scenarios
    - bins: (optional) np.array of shape (N_bins, 2) -> the first column contains the wind speed intervals and the second 
        column contains the corresponding normalized power
        
    Returns:
    - norm_power: np.array of the same shape as wind_speed containing the normalized power values
    """
    if bins is None:
        ws_bin_processed = np.array(pd.read_csv('method_of_bins.csv', usecols = [0,1]))
    else:
        ws_bin_processed = bins
    
    norm_power = np.zeros_like(wind_speed)

    for j in range(ws_bin_processed.shape[0]-1):
        index_true_STH = (wind_speed>=ws_bin_processed[j,0]) & (
            wind_speed<ws_bin_processed[j+1,0])
        norm_power[index_true_STH] = ws_bin_processed[j,1]
    
    return norm_power 


# In[8]:


def get_mission_time_STH(wind_speed,
                        wave_height,
                        vessel_max_wave_height,
                        vessel_max_wind_speed,
                        tau,
                        tR=5,
                        tD=21
                        ):
    
    _,N_t,N_S = wind_speed.shape
    
    # Hourly accessibility
    access = np.zeros_like(wind_speed)
    access[(wind_speed<vessel_max_wind_speed) & (
        np.repeat(wave_height.reshape(-1,1,N_S),N_t,axis=1)<vessel_max_wave_height)] = 1
    access2 = access.reshape(-1,24,N_t,N_S)
    access2[:,:tR,:,:] = 0 #sunrise constraint
    access2[:,tD:,:,:] = 0 #sunset constraint
    #print(np.arange(1,25))
    #print(access2[:,:,2,0])
    
    # Calculate operation times
    op_time = np.zeros((wind_speed.shape), dtype=int)
    op_time2 = op_time.reshape(-1,24,N_t,N_S)
    
    
    for i in range(N_t):
        for t in range(24):
            temp = op_time2[:,t,i,:]
            op_time2[:,t,i,:] = (np.cumsum(access2[:,t:,i,:],1)>=tau[i]).argmax(1)
            temp[temp==0] = (23-t+tau[i]-np.sum(access2[:,t:,i,:],1))[temp==0]
    
    op_time2+=1
    
    return op_time2[0,:,:,:], access2[0,:,:,:]


# In[9]:


def data_loader(ws_STH,  
                ws_LTH,
                wh_STH,
                wh_LTH,
                ep_STH,
                ep_LTH, 
                ws_err_hist,
                wh_err_hist,
                ep_err_hist,
                N_S,
                max_wind, 
                max_wave,
                rle_mean,
                rle_std,
                tau,
                rle_dist='weibull',
                tR=5, 
                tD=21,
                hist_len=5,
                random_seed=1,
                sim_day_ahead=False,
                BESN=False,
                NAIVE=False,
                DGP=False
               ):
    """
    A function used to prepare the necessary stochastic inputs of STOCHOS using forecast values. In this model we use the 
    true hourly data to generate scenarios for the STH and the LTH. Future versions will require daily forecasts using 
    machine learning methods to generate scenarios.
    
    Inputs:
    - ws_STH: np.array of shape (24,N_t) -> Hourly wind speed data for each wind turbine in the STH
    - ws_LTH: np.array of shape (dJ,N_t) -> Daily average wind speed data for each wind turbine in the LTH
    - wh_STH: np.array of shape (24,) -> Hourly wave height data for STH
    - wh_LTH: np.array of shape (dJ,) -> Daily average wave height data for LTH
    - ep_STH: np.array of shape (24,) -> Hourly electricity prices data for STH
    - ep_LTH: np.array of shape (dJ,) -> Daily average electricity price data for LTH
    - ws_err_hist: np.array of shape (hist_len,N_t) -> Historical hourly wind speed forecast error data for each 
        turbine
    - wh_err_hist: np.array of shape (hist_len,) -> Historical hourly wave height forecast error data
    - ep_err_hist: np.array of shape (hist_len,) -> Historical hourly electricity price forecast error data
    - N_S: integer -> Number of scenarios to generate
    - max_wind: float -> Safety threshold for maximum wind speed
    - max_wave: floar -> Safety threshold for maximum wave height
    - rle_mean: np.array of shape (N_t,) -> Mean of residual life estimates for each wind turbine
    - rle_std: float or np.array of floats of shape (N_t,) -> RLE forecast STD used in random scenario generator
    - tau: integer np.array of shape (N_t,) -> Workhours required to complete each maintenance task
    - rle_dist: (optional) str -> the distibution to be used for the scenario generation. Valid inputs are 'normal' and 'weibull'.
        If 'weibull' is selected, the rle_std is used as the shape parameter, and rle_mean as the scale.
    - tR: (optional) integer -> Time of first sunlight in 24-hour basis (default=5) 
    - tD: (optional) integer -> Time of last sunlight in 24-hour basis (default=21) 
    - hist_len (optional) integer -> Custom history length for scenario generation in days (default=5)
    - random_seed: (optional) integer -> Seed used for reproducibility of scenario generation (default=1)
    - sim_day_ahead: (optional) bool -> Simulate the day ahead using a predefined schedule and true data/no scenarios 
        (default=False)
        
    Returns:
    - norm_power: np.array of shape (24, N_t, N_S) -> hourly normalized power scenarios of the wind turbines in the STH
    - norm_powerL: np.array of shape (dJ, N_t, S) -> daily normalized power scenarios of the wind turbines in the LTH
    - el_price: np.array of shape (24, N_S) -> hourly electricity price scenarios in the STH [$/MWh]
    - el_priceL: np.array of shape (dJ, N_S) -> daily electricity price scenarios in the LTH [$/MWh]
    - op_time: np.array of shape (24, N_t, N_S) -> hourly operation time scenarios for turbine maintenance [hours]
    - op_timeL: np.array of shape (dJ, N_t, N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - rle: np.array of shape (N_t, N_S) -> Scenarios for RLE
    - access: np.array of shape (24, N_t, N_S) -> STH accessibility
    - access_LTH: np.array of shape (dJ, N_t, N_S) -> LTH accessibility
    - ws_STH_scenarios: np.array of shape (24, N_t, N_S) -> STH wind speed scenarios
    - wh_STH_scenarios: np.array of shape (24, N_S) -> STH wave height scenarios 
    - ws_LTH_scenarios: np.array of shape (dJ, N_t, N_S) -> LTH wind speed scenarios 
    - wh_LTH_scenarios: np.array of shape (dJ, N_S) -> LTH wave height scenarios
    """
    dJ, N_t = ws_LTH.shape
    
    t1=time()
    
    if (sim_day_ahead) | (N_S==1):
        ws_STH_scenarios = ws_STH[:,:,np.newaxis]
        wh_STH_scenarios = wh_STH[:,np.newaxis]
        ws_LTH_scenarios = ws_LTH[:,:,np.newaxis]
        wh_LTH_scenarios = wh_LTH[:,np.newaxis]
        
        el_price = ep_STH[:,np.newaxis]
        el_priceL = ep_LTH[:,np.newaxis]

    else:
        ws_STH_scenarios = np.repeat(ws_STH[:,:,np.newaxis], N_S, -1,) 
        ws_LTH_scenarios = np.repeat(ws_LTH[:,:,np.newaxis], N_S, -1,) 
        
        wh_STH_scenarios = np.repeat(wh_STH[:,np.newaxis], N_S, -1,) 
        wh_LTH_scenarios = np.repeat(wh_LTH[:,np.newaxis], N_S, -1,) 
        
        el_price = np.repeat(ep_STH[:,np.newaxis], N_S, -1,) 
        el_priceL = np.repeat( ep_LTH[:,np.newaxis], N_S, -1,) 
        
        if NAIVE:
            np.random.seed(random_seed)
            for i in range(N_t):
                ws_STH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ws_err_hist[:,i].std()),size=(24,N_S))
                ws_LTH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),ws_err_hist[:,i].std()),size=(dJ,N_S))
            
                wh_STH_scenarios -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),wh_err_hist.std()),size=(24,N_S))
                wh_LTH_scenarios -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),wh_err_hist.std()),size=(dJ,N_S))
                
                el_price -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ep_err_hist.std()),size=(24,N_S))
                el_priceL -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),ep_err_hist.std()),size=(dJ,N_S))
        else:
            for i in range(N_t):
                if not(BESN): ws_STH_scenarios[:,i,:] -= error_scenario_gen(ws_err_hist[:,i], 24, N_S, 
                      24*hist_len, random_seed, DGP=DGP)
                ws_LTH_scenarios[:,i,:] -= error_scenario_gen(ws_err_hist[:,i].reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, LDS=True, DGP=DGP)[1:,:] #discard the day-ahead
        
            if not(BESN): wh_STH_scenarios -= error_scenario_gen(wh_err_hist, 24, N_S, 24*hist_len, 
                  random_seed, LDS=True, DGP=DGP)
            wh_LTH_scenarios -= error_scenario_gen(wh_err_hist.reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, LDS=True, DGP=DGP)[1:,:] #discard the day-ahead
            
            el_price -= error_scenario_gen(ep_err_hist, 24, N_S, 24*hist_len, random_seed, DGP=DGP)
            el_priceL -= error_scenario_gen(ep_err_hist.reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, DGP=DGP)[1:,:] #discard the day-ahead
        
        ws_STH_scenarios[ws_STH_scenarios<0.5]=0.5
        wh_STH_scenarios[wh_STH_scenarios<0.1]=0.1
        ws_LTH_scenarios[ws_LTH_scenarios<0.5]=0.5
        wh_LTH_scenarios[wh_LTH_scenarios<0.1]=0.1
        

    norm_power = binning_method(ws_STH_scenarios)
    norm_powerL = binning_method(ws_LTH_scenarios)
    
    op_time, access = get_mission_time_STH(
            ws_STH_scenarios,
            wh_STH_scenarios,
            max_wave,
            max_wind,
            tau,
            tR,
            tD)
    

    op_timeL, access_LTH = get_mission_time_LTH(ws_LTH_scenarios, 
                                                wh_LTH_scenarios, 
                                                tau)
        
    if (sim_day_ahead) | ((N_S==1) & (not(DGP))):
        rle = rle_mean[:,np.newaxis]
    else:
        if DGP:
            rle = (rle_mean*gamma(1+1/rle_std)).reshape(-1,1)
        else:
            rle = rle_scenario_gen(rle_mean, rle_std, N_S, rle_dist, random_seed)
        
    print('Data prepared in ', round(time()-t1, 4), ' sec')
    
    return norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access, access_LTH, ws_STH_scenarios, wh_STH_scenarios, ws_LTH_scenarios, wh_LTH_scenarios


# In[10]:


def get_var_from_model(model_dict, var_name, STH_len=24, LTH_len=19, N_t=5):
    """
    Function to help extract the model variables stored as dictionary items into numpy arrays.
    
    Inputs:
    - model_dict: A dictionary whose keys are the single variable names of the model in string format 
        ('var_name[time_idx,wt_idx,scenario_idx]') that stores the corresponding variable value.
    - var_name: A string with the name of the block of variables. 
    - STH_len: (optional; Default=24) The length of the STH.
    - LTH_len: (optional; Default=19) The length of the LTH.
    - N_t: (optional; Default=5) The number of wind turbines considered.
    
    Returns:
    - var_value_array: A numpy array with the values of the querried variable.
    - var_name_array: A numpy array with the names of each single variable.
    """
    var_name_list = []
    var_value_list = []
    for var, value in model_dict.items():
        if var_name+'[' in var:
            var_name_list.append(var)
            var_value_list.append(value)
    
    var_name_array = np.array(var_name_list)
    var_value_array = np.array(var_value_list)
    
    if '[d' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,N_t,-1)
                var_value_array = var_value_array.reshape(LTH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len,N_t)
                var_value_array = var_value_array.reshape(LTH_len,N_t)
        else:
            if ',s]' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,-1)
                var_value_array = var_value_array.reshape(LTH_len,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len)
                var_value_array = var_value_array.reshape(LTH_len)
                
    elif '[t' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,N_t,-1)
                var_value_array = var_value_array.reshape(STH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len,N_t)
                var_value_array = var_value_array.reshape(STH_len,N_t)
        else:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,-1)
                var_value_array = var_value_array.reshape(STH_len,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len)
                var_value_array = var_value_array.reshape(STH_len)
    
    elif '[wt' in var_name_array[0]:
        var_name_array = var_name_array.reshape(N_t,-1)
        var_value_array = var_value_array.reshape(N_t,-1)
    else:
        var_name_array = var_name_array.reshape(-1,1)
        var_value_array = var_value_array.reshape(-1,1)
    
    if var_name=='v':
        return np.array(model_dict['v']), np.array(['v'])
    
    else:
        return var_value_array, var_name_array


# In[14]:


def RH_iter_solver( wind_speed_true,
                    wave_height_true,
                    wind_speed_forecast,
                    wave_height_forecast,
                    N_S,
                    Din,
                    max_wind, 
                    max_wave,
                    el_prices_true,
                    el_prices_forecast,
                    rle_true,
                    rle_forecast_mean,
                    rle_forecast_std,
                    tau,
                    rho,
                    theta,
                    rle_dist='weibull',
                    tR=5, 
                    tD=21,
                    dJ=19,
                    hist_len=5,
                    random_seed=1,
                    max_iter=None,
                    mip_gap=0.001,
                    CMS=False,
                    AIMS=False,
                    BESN=False,
                    TBS=False,
                    NAIVE=False,
                    DGP=False):
    """
    Function that uses the rolling horizon iterative algorithm that gets the output of the stochastic model and uses it as
    input to a simulation model that uses true data to get the true schedule cost.
    
    Inputs:
    -wind_speed_true: np.array of shape (obs_dim, N_t) -> The true wind speed profile that will be used in the simulation 
         of the day-ahead
    -wave_height_true: np.array of shape (obs_dim, N_t) -> The true wave height profile that will be used in the simulation 
         of the day-ahead
    -wind_speed_forecast: np.array of shape (obs_dim, N_t) -> The forecasted wind speed profile that will be used in the 
        stochastic program
    -wave_height_forecast: np.array of shape (obs_dim, N_t) -> The forecasted wave height profile that will be used in the 
        stochastic program
    -N_S: int -> Number of scenarios to generate
    -Din: int -> Day of the dataset that will be the first day
    -wind_scenario_std: float or np.array of (time_dim, N_t) -> The STH of the wind speed point forecasts to be used for the 
        scenario generation
    -wave_scenario_std: : float or np.array of (time_dim, N_t) -> The STH of the wave height point forecasts to be used for the 
        scenario generation
    -max_wind: float -> Maximum wind speed tolerance for vessel/crew operation 
    -max_wave: float -> Maximum wave height tolerance for vessel/crew operation 
    -el_prices_true: np.array of shape (time_dim, ) -> The true electricity price profile that will be used in the simulation 
         of the day-ahead
    -el_prices_forecast: np.array of shape (time_dim, ) -> The forecasted electricity price profile that will be used in the 
        stochastic program
    -rle_true: np.array of shape (N_t, ) -> The true RLEs that will be used in the simulation of the day-ahead (in days)
    -rle_forecast: np.array of shape (N_t, ) -> The forecasted RLEs that will be used in the stochastic program (in days)
    -rle_std: np.array of shape (N_t, ) -> The RLE STDs that will be used in the stochastic program (in days)
    -tau: int np.array of shape (N_t, ) -> Time to complete each task
    -rho: bool np.array of shape (N_t, ) -> Maintenance was previously initiated
    -theta: bool np.array of shape (N_t, ) -> Maintenance is required
    
    
    Optional:
    - rle_dist: (default='weibull') str -> the distibution to be used for the scenario generation. Valid inputs are 'normal' and 'weibull'.
        If 'weibull' is selected, the rle_std is used as the shape parameter, and rle_mean as the scale.
    - tR: int (default=5) -> Time of first daylight 
    - tD: int (default=21) -> Time of last daylight 
    - dJ: int (default=19) -> LTH length in days
    - hist_len: int (default=5) -> History length used in GPR for scenario generation in days
    - random_seed: int (default=1) -> Random seed
    - max_iter: int (default=None) -> Maximum number of iterations. If None is selected, the horizon rolls until all wind
        turbines have a completed maintenance task
    - mip_gap: float (default=0.001) -> Optimality maximum relative gap for the stochastic solution
    
    Returns:
    -output: A dictionary with items:
    'stochastic_sol': a list whose items are the dict outputs of STOCHOS at each horizon roll
    'stochastic_input': a tuple whose elements are lists of the stochastic inputs to STOCHOS at each horizon roll. Each list has
        inputs with the following order:
            0. norm_power
            1. norm_powerL
            2. el_price
            3. el_priceL
            4. op_time
            5. op_timeL
            6. access
            7. ws_STH
            8. wh_STH
            9. rle
            10. theta
            11. rho
    'simulation': a list whose items are the dict outputs of the simulation at each horizon roll using the STH schedule
        of 'stochastic_sol'
    'true_input': a tuple whose elements are lists of the true inputs to the simulation model at each horizon roll, following
        the same order as the 'stochastic_input'
    'time_per_roll': a list with the optimization time of each horizon roll
    'total_hourly_sched_time': a float with the total optimization time of all rolls
    """
    
    N_t = wind_speed_true.shape[1]
    
    stochastic_input = []
    stochastic_output = []
    
    true_input = []
    true_output = []
    one_iteration_time = []
    
    rle_true = rle_true.copy()
    rle_forecast_mean = rle_forecast_mean.copy()
    tau = tau.copy()
    rho = rho.copy()
    theta = theta.copy()
    
    t0 = time()

    run=0
    while np.any(theta>0) | np.any(rho>0):

        t1 = time()
        
        wind_speed_error_hist = wind_speed_forecast[:(Din+run)*24,:]-wind_speed_true[:(Din+run)*24,:]
        wave_height_error_hist = wave_height_forecast[:(Din+run)*24]-wave_height_true[:(Din+run)*24]
        el_price_error_hist = el_prices_forecast[:(Din+run)*24]-el_prices_true[:(Din+run)*24]
        
        wsf_STH = wind_speed_forecast[24*(Din+run):24*(Din+run+1),:]
        whf_STH = wave_height_forecast[24*(Din+run):24*(Din+run+1)]
        wsf_LTH = wind_speed_forecast[24*(Din+run+1):24*(Din+run+dJ+1),:].reshape(-1,24,N_t).mean(1)
        whf_LTH = wave_height_forecast[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        
        wst_STH = wind_speed_true[24*(Din+run):24*(Din+run+1),:]
        wht_STH = wave_height_true[24*(Din+run):24*(Din+run+1)]
        wst_LTH = wind_speed_true[24*(Din+run+1):24*(Din+run+dJ+1),:].reshape(-1,24,N_t).mean(1)
        wht_LTH = wave_height_true[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        
        epf_STH = el_prices_forecast[24*(Din+run):24*(Din+run+1)]
        ept_STH = el_prices_true[24*(Din+run):24*(Din+run+1)]
        epf_LTH = el_prices_forecast[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        ept_LTH = el_prices_true[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        

        (norm_power, 
         norm_powerL, 
         el_price, 
         el_priceL, 
         op_time, 
         op_timeL, 
         rle,
         access, access_LTH ,ws_STH,wh_STH,_,_) = data_loader(
                 wsf_STH, 
                 wsf_LTH, 
                 whf_STH,
                 whf_LTH,
                 epf_STH,
                 epf_LTH,
                 wind_speed_error_hist,
                 wave_height_error_hist,
                 el_price_error_hist,
                 N_S, 
                 max_wind, 
                 max_wave,
                 rle_forecast_mean,
                 rle_forecast_std,
                 tau,
                 rle_dist,
                 tR, 
                 tD,
                 hist_len,
                 random_seed,
                 sim_day_ahead=False,
                 BESN=BESN,
                 NAIVE=NAIVE,
                 DGP=DGP)
        
        
        if AIMS:
            op_time = np.repeat(np.repeat(tau[np.newaxis,:,np.newaxis],24,0),N_S,2)
            op_timeL = np.repeat(np.repeat(tau[np.newaxis,:,np.newaxis],dJ,0),N_S,2)
            access_LTH = np.ones((dJ,N_t,N_S))
        
        if BESN:
            #el_price = np.ones((24,N_S))*60*1.15 #Besnard 2011 electricity price = 60 euros/MWh
            #el_priceL = np.ones((dJ,N_S))*60*1.15 #Besnard 2011 electricity price = 60 euros/MWh
            #el_price = np.repeat(epf_STH.copy()[:,np.newaxis],N_S,1)
            #el_priceL = np.repeat(epf_LTH.copy()[:,np.newaxis],N_S,1)
            el_price = np.ones((24,N_S))*np.mean(el_prices_forecast) 
            el_priceL = np.ones((dJ,N_S))*np.mean(el_prices_forecast) 
            rle = np.repeat(rle_forecast_mean[:,np.newaxis],N_S,1)
        
        stochastic_input.append((norm_power.copy(),norm_powerL.copy(),el_price.copy(),el_priceL.copy(),op_time.copy(),
                                 op_timeL.copy(),access.copy(), ws_STH.copy(), wh_STH.copy(), access_LTH.copy(),
                                 rle.copy(),theta.copy(),rho.copy()))
        
        stochastic_output.append(STOCHOS(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access_LTH, theta,
                                         rho, curtailment=0, scalars=None, STH_sched=None, return_all_vars=True,
                                         rel_gap = mip_gap, CMS=CMS, TBS=TBS))
        
        if stochastic_output[-1]==None: break
        
        stoch_STH_sched = stochastic_output[run]['STH_sched']

        (norm_power, 
         norm_powerL, 
         el_price, 
         el_priceL, 
         op_time, 
         op_timeL, 
         rle,
         access, access_LTH,ws_STH,wh_STH,_,_) = data_loader(
                 wst_STH, 
                 wst_LTH, 
                 wht_STH,
                 wht_LTH,
                 ept_STH,
                 ept_LTH,
                 wind_speed_error_hist,
                 wave_height_error_hist,
                 el_price_error_hist,
                 N_S, 
                 max_wind, 
                 max_wave,
                 rle_true,
                 0,
                 tau,
                 rle_dist,
                 tR, 
                 tD,
                 hist_len,
                 random_seed,
                 sim_day_ahead=True)
        
        true_input.append((norm_power.copy(),norm_powerL.copy(),el_price.copy(),el_priceL.copy(),op_time.copy(),
                           op_timeL.copy(),access.copy(), ws_STH.copy(), wh_STH.copy(), access_LTH.copy(),
                           rle.copy(),theta.copy(),rho.copy()))
        
        true_output.append(STOCHOS(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access_LTH, theta,
                                   rho, curtailment=0, scalars=None, STH_sched=stoch_STH_sched, return_all_vars=True,
                                   rel_gap = 0.0))
        
        if true_output[-1]==None: break
        
        #decrease rle means by 1 day, as we move 1 day forward:
        rle_true -= 1
        rle_true[rle_true<0]=0
        
        rle_forecast_mean -= 1
        rle_forecast_mean[rle_forecast_mean<0]=0
        
        #if the true rle is zero, set the mean value of the forecast to zero
        rle_forecast_mean[rle_true==0] = 0
        
        #if the rle mean of the forecast says 0 days but the turbine is still working,
        #then correct the forecast mean, by setting it to 1 day:
        rle_forecast_mean[(rle_forecast_mean==0) & (rle_true>0)] = 1
        
        
        
        true_STH = np.array(true_output[run]['STH_sched'])
        
        wind_speed_error_hist = np.append(wind_speed_error_hist,
                                          wsf_STH-wst_STH, 0)
        wave_height_error_hist = np.append(wave_height_error_hist,
                                          whf_STH-wht_STH, 0)
        
        
        #check for unfinished tasks, or completed tasks in the STH:
        for i in range(N_t):
            if (np.any(true_STH[:,i])>0):
                if true_output[run]['remaining_maint_hrs'][i] == 0:
                    theta[i]=0
                    rho[i]=0
                else:
                    theta[i]=1
                    rho[i]=1
                    tau[i]=true_output[run]['remaining_maint_hrs'][i,0]

        print('theta:', theta)
        print('rho:', rho)
        print('tau:', tau)

        run+=1
        one_iteration_time.append(time()-t1)
        
        if run==max_iter: break
        
    return {  'stochastic_sol':stochastic_output,
              'stochastic_input': stochastic_input,
              'simulation': true_output,
              'true_input':true_input,
              'time_per_roll': one_iteration_time,
              'total_hourly_sched_time': time()-t0 }


# In[23]:


def show_full_schedule(solution, return_schedule = False):
    """
    Generates a Gantt-chart for the STH schedule from all runs. 
    True residual lives are displayed by orange vertical lines.
    RLE forecast scenarios are displayed by red vertical lines.
    If return_schedule is set to true, it also returns the full hourly schedule.
    """    
    schedule = np.array(solution['simulation'][0]['STH_sched'])
    op_time = solution['true_input'][0][4]

    for i in range(1,len(solution['time_per_roll'])):
        schedule = np.append(schedule,np.array(solution['simulation'][i]['STH_sched']), 0)
        op_time = np.append(op_time, solution['true_input'][i][4],0)

    rle_true = solution['true_input'][0][-3]
    rle_stoch = solution['stochastic_input'][0][-3]

    
    N_t, N_S = rle_stoch.shape
    
    trns = 1 if N_S==1 else 0.3

    x = np.arange(schedule.shape[0])#.reshape(-1,1)#, N_t, 1)
    rle_true = 24*rle_true.copy()
    rle_stoch = 24*rle_stoch.copy()
    
    #print('schedule.shape',schedule.shape)
    #('x.shape',x.shape)
    #print('op_time.shape',op_time.shape)
    
    for i in range(N_t):
        plt.vlines(rle_true[i],i+1-0.5,i+1+0.5, color='orange', linewidth=3,alpha=1)
        plt.hlines(np.full(schedule.shape[0], i+1),x*schedule[:,i],(x+op_time[:,i,0])*schedule[:,i], 
                           color='black', linewidth=15, alpha=1)
        plt.vlines(rle_stoch[i,:],np.full(N_S, i+1-0.5),np.full(N_S, i+1+0.5), color='red', linewidth=2,alpha=trns)

    if return_schedule: return schedule


#%%
    
def show_full_revenue(solution, return_revenue=False, patch_LTH_revenue=False, R=12):
    """
    Returns the full hourly revenue profile of the expected revenue, summed accross all wind turbines.
    Real revenue is displayed with a red line, and scenarios with grey lines.
    R: float -> The rated power of the wind turbines.
    If return_revenue is set to True, it also returns a pair with the real hourly revenue and the revenue 
    forecast scenarios as np arrays.
    If patch_LTH_revenue is set to True, it also returnse the LTH revenue from the last roll, 
    with a daily resolution
    """
    true_revenue =  np.array(solution['true_input'][0][2]*solution['true_input'][0][0].sum(1)*R)
    stoch_revenue = np.array(solution['stochastic_input'][0][2]*solution['stochastic_input'][0][0].sum(1)*R)
    
    dJ, N_S = stoch_revenue.shape
    
    
    
    for i in range(1,len(solution['time_per_roll'])):
        true_revenue = np.append(true_revenue,np.array(
                solution['true_input'][i][2]*solution['true_input'][i][0].sum(1)*R), 0)
        stoch_revenue = np.append(stoch_revenue,np.array(
                solution['stochastic_input'][i][2]*solution['stochastic_input'][i][0].sum(1)*R), 0)
    
    if patch_LTH_revenue:
        true_LTH_rev = np.repeat((solution['true_input'][-1][3]*solution['true_input'][-1][1].sum(1)*R
                                  )[:,:,np.newaxis],24,-1).reshape(-1,1)
        stoch_LTH_rev = np.repeat((solution['stochastic_input'][-1][3]*solution['stochastic_input'][-1][1].sum(1)*R
                                   )[:,:,np.newaxis],24,-1).transpose(0,2,1).reshape(-1,N_S)
        
        true_revenue = np.append(true_revenue, true_LTH_rev, 0)
        stoch_revenue = np.append(stoch_revenue, stoch_LTH_rev, 0)
    
    
    plt.plot(true_revenue[:,0], color='red', linewidth=2)
    plt.plot(stoch_revenue, color='grey', alpha=0.5)
    plt.plot(true_revenue[:,0], color='red', linewidth=2)
    plt.ylabel('Total revenue ($)')
    
    if return_revenue: return (true_revenue, stoch_revenue)


# In[24]:


def show_LTH_schedule(solution, roll=0 , return_schedule=False):
    """
    Generates a Gantt-chart for the LTH schedule obtained at a roll. 
    True residual lives are displayed by orange vertical lines.
    RLE forecast scenarios are displayed by red vertical lines.
    If return_schedule is set to true, it also returns the LTH schedule for this roll.
    """    
    x = solution['stochastic_sol'][roll]['LTH_sched'].copy()
    op_time = solution['stochastic_input'][roll][5]
    rle = solution['stochastic_input'][roll][-3]
    rle_true = solution['true_input'][roll][-3]
    
    dJ, N_t, N_S = op_time.shape
        
    for i in range(N_t):
        x[:,i,:] = x[:,i,:]*(i+1)
        plt.vlines(rle[i,:]-1,np.full(N_S,i+1-0.5),np.full(N_S,i+1+0.5), color='red', linewidth=1,alpha=0.2)
        plt.vlines(rle[i,:].mean()-1,i+1-0.5,i+1+0.5, color='red', linewidth=2,alpha=1)
        plt.vlines(rle_true[i]-1,i+1-0.5,i+1+0.5, color='orange', linewidth=2,alpha=1)
        for j in range(dJ):
            plt.hlines(np.full(N_S,i+1),j*x[j,i,:]/(i+1),j*x[j,i,:]/(i+1)+x[j,i,:]*op_time[j,i,:]/24, 
                       color='black', linewidth=15, alpha=0.1)
    plt.ylim([0.5,N_t+0.5])
    plt.yticks([i for i in range(1,N_t+1)], ['WT'+str(i) for i in range(1,N_t+1)])
    plt.xticks([i for i in range(dJ)],[i for i in range(1,dJ+1)])
    plt.xlabel('Day of the LTH')
            
    if return_schedule: return x



# In[ ]:

def show_weather_profile(wind_speed_true, 
                         wave_height_true,
                         wind_speed_forecast, 
                         wave_height_forecast,
                         Din, 
                         N_S,
                         turbine = 0,
                         horizon_len=5,
                         max_wind=15, 
                         max_wave=1.5):
    
    N_t = wind_speed_true.shape[1]
    ws_scenarios=np.zeros((24*horizon_len,N_t,N_S))
    wh_scenarios=np.zeros((24*horizon_len,N_S))
    access = np.zeros((24*horizon_len, N_t, N_S))
    
    for run in range(horizon_len):
        wind_speed_error_hist = wind_speed_forecast[:(Din+run)*24,:]-wind_speed_true[:(Din+run)*24,:]
        wave_height_error_hist = wave_height_forecast[:(Din+run)*24]-wave_height_true[:(Din+run)*24]
        wsf_STH = wind_speed_forecast[24*(Din+run):24*(Din+run+1),:]
        whf_STH = wave_height_forecast[24*(Din+run):24*(Din+run+1)]
            
        _, _, _, _, _, _, _, acc, wss, whs, _, _ = data_loader( ws_STH=wsf_STH,  
                                                                ws_LTH=np.zeros((1,N_t)),
                                                                wh_STH=whf_STH,
                                                                wh_LTH=np.zeros((1,)),
                                                                ep_STH=np.zeros((24,)),
                                                                ep_LTH=np.zeros((1,)), 
                                                                ws_err_hist=wind_speed_error_hist,
                                                                wh_err_hist=wave_height_error_hist,
                                                                ep_err_hist=np.zeros(100*24),
                                                                N_S=N_S,
                                                                max_wind=max_wind, 
                                                                max_wave=max_wave,
                                                                rle_mean=np.ones(N_t),
                                                                rle_std=np.zeros(N_t),
                                                                tau=np.ones(N_t),
                                                                rle_dist='weibull',
                                                                tR=5, 
                                                                tD=21,
                                                                random_seed=1,
                                                                sim_day_ahead=False
                                                               )
    
        ws_scenarios[24*run:24*(run+1),:,:] = wss.copy()
        wh_scenarios[24*run:24*(run+1),:] = whs.copy()
        access[24*run:24*(run+1),:,:] = acc.copy()
        
    plt.plot(ws_scenarios[:,turbine,:], color='grey', alpha=0.25)
    plt.plot(wind_speed_forecast[Din*24:(Din+horizon_len)*24,turbine], color='grey', linewidth=2)
    plt.plot(wh_scenarios[:,:], color=[0.7, 0.8, 1], alpha=0.25)
    plt.plot(wave_height_forecast[Din*24:(Din+horizon_len)*24], color='cyan', linewidth=2)
    plt.plot(wind_speed_true[Din*24:(Din+horizon_len)*24,turbine], color='black',linewidth=2)
    plt.plot(wave_height_true[Din*24:(Din+horizon_len)*24], color='blue',linewidth=2)
    plt.hlines(max_wind, 0, 24*horizon_len, color='black', linestyle='--')
    plt.hlines(max_wave, 0, 24*horizon_len, color='blue', linestyle='--')
    
    return access
        

def make_light(sol_dict):
    for i in sol_dict.keys():
        for j in range(len(sol_dict[i]['time_per_roll'])):
            sol_dict[i]['stochastic_sol'][j]['model'] = None
            sol_dict[i]['simulation'][j]['model'] = None
    return sol_dict


