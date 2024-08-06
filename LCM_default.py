import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from gym import Env
from gymnasium.envs.registration import register
from gymnasium.spaces import Discrete, Box

#################################################################################################
#                                                                                               #
# Set default options for the LCM.                                                              #
# a: hyperparameter of the Beta prior for the Bernoulli likelihood                              #
# b: hyperparameter of the Beta prior for the Bernoulli likelihood                              #
# alpha: oncentrate parameter of the Chinese Restaurant Process                                 #
# stickiness: stickiness parameter of the Chinese Restaurant Process                            #
# K: maximum number of latent causes                                                            #
# M: number of particles, if M=1, local MAP estimate is computed                                #
#                                                                                               #
# LCM_opts & LCM_infer are adapted from https://github.com/sjgershm/LCM/blob/master/LCM_infer.m #
#################################################################################################

def LCM_opts(opts=None):
 
    def_opts = {
        'a': 1,
        'b': 1,
        'alpha': 1,
        'stickiness': 0,
        'K': 10,
        'M': 1
    }
    
    # If no options are provided, use the default options.
    if not opts:
        opts = def_opts
    else:
        # Fill in the missing options with the default options.
        for key in def_opts.keys():
            if key not in opts or not opts[key]:
                opts[key] = def_opts[key]
    
    # Set values to be non-negative.
    opts['a'] = max(opts['a'], 0)
    opts['b'] = max(opts['b'], 0)
    opts['alpha'] = max(opts['alpha'], 0)
    opts['stickiness'] = max(opts['stickiness'], 0)
    
    return opts

# LCM inference function.
def LCM_infer(X, opts=None):

    if not opts:
        opts = {}

    # Option defaults.
    opts = LCM_opts(opts)
    M = opts['M']
    a = opts['a']
    b = opts['b']
    results = {}
    results['opts'] = opts

    if opts['alpha'] == 0:
        K = 1
    else:
        K = opts['K']
    
    # Initialize the posterior and posterior for the zero state.
    post = np.zeros(K); post[0] = 1
    post_zero = np.zeros((M,K)); post_zero[0][0] = 1

    T, D = X.shape
    N = np.zeros((M,K,D))
    B = np.zeros((M,K,D))
    Nk = np.zeros((M,K))

    results['post'] = np.hstack((np.ones((T,1)), np.zeros((T,K-1))))
    results['V'] = np.zeros((T,1))
    z = np.ones((M, 1))
    z = z.astype(int)

    for t in range(T):

        lkhd = N.copy()
        lkhd[:,:,X[t,:]==0] = B[:,:,X[t,:]==0]
        fraction_lkhd = lkhd + a
        fraction_Nk = Nk + a + b

        # Compute likelihoods.
        for d in range(D):
            lkhd[:,:,d] = fraction_lkhd[:,:,d] / fraction_Nk

        if opts['alpha'] > 0:
            prior = Nk.copy()

            # Add stickiness to the prior.
            for m in range(M):
                prior[m, z[m]] += opts['stickiness']
                #prior[m,np.where(prior[m,:]==0)[0][0]] = opts['alpha']
                ### this was the old line, it fails for alpha values > 2 #### now fixed with zero_idx
                zero_idx = np.where(prior[m,:]==0)[0]
                if zero_idx.size > 0:
                    prior[m, zero_idx[0]] = opts['alpha']

            prior = prior / prior.sum(axis=1, keepdims=True)

            # Compute the posterior probabilities.
            post = np.multiply(prior, np.squeeze(np.prod(lkhd[:, :, 1:D], axis=2)))
            post_zero = post / post.sum(axis=1, keepdims=True)
            post = post * lkhd[:, :, 0]
            post = post / np.sum(post)

        results['post'][t,:] = np.mean(post / post.sum(axis=1, keepdims=True), axis=0)

        pUS = (N[:, :, 0] + a) / (Nk + a + b)

        results['V'][t, 0] = np.dot(post_zero.flatten(), pUS.flatten()) / M

        # Sample new particles from X.
        # Convert the logical array to 0s and 1s.
        x1 = X[t,:] == 1
        x0 = X[t,:] == 0

        if M == 1:
            z = np.argmax(post, axis=1)

            # Update the matrices.
            Nk[0, z] += 1
            N[0, z, x1] += 1
            B[0, z, x0] += 1

        else:
            Nk_old = Nk.copy()
            N_old = N.copy()
            B_old = B.copy()

            for m in range(M):
                row = np.min(np.argwhere(np.random.uniform(0,1) < np.cumsum(np.sum(post, axis=1))))

                Nk[m, :] = Nk_old[row, :]
                N[m, :, :] = N_old[row, :, :]
                B[m, :, :] = B_old[row, :, :]

                col = np.min(np.argwhere(np.random.uniform(0,1) < np.cumsum(post[row, :] / np.sum(post[row, :]))))

                Nk[m, col] += 1
                N[m, col, x1] += 1
                B[m, col, x0] += 1
   
    return {'V': results['V'], 'post': results['post'], 'opts': opts}

###########################################################################################
# LCM inference function extended to include the Rescorla-Wagner model.                   #
# Adapted from https://github.com/sjgershm/memory-modification/blob/master/imm_localmap.m #
###########################################################################################

def LCM_infer_rw(X, opts):
    
    zp_save = []
    w_save = []
    p_save = []
    w_before_r = np.array([])
    post_before_r = np.array([])
    T = X.shape[0]

    # Time vals.
    time = X[:, 0]

    # Reward vals.
    r = X[:, 1]

    # Cues (columns CS1 and CS2)
    X = X[:, 2:]

    D = X.shape[1]

    if opts['c_alpha'] == 1:
        opts['c_alpha'] = opts['c_alpha'] * np.ones((T,1))
    if ['eta'] == 1:
        opts['eta'] = opts['eta'] * np.ones((T, 1))
    
    opts['c_alpha'] = np.full((T, 1), opts['c_alpha'])
    opts['eta'] = np.full((T, 1), opts['eta'])

    psi = opts['psi'] * np.ones((T,1))
    
    Z = np.zeros((T, opts['K']))
    V = np.zeros((T, 1))
    W = np.zeros((D, opts['K'])) + opts['w0']

    # Construct the distance matrix.
    Dist = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            Dist[i, j] = np.absolute(time[i] - time[j])
        
    # THERE IS A DIVIDE BY ZERO WARNING HERE DUE TO THE DIST ARRAY
    S = Dist ** (-opts['g'])

    iter_vals = []
    t_vals = []
  
    # Compute inference.
    for t in range(1, T + 1):
        # Determine how many EM iterations to perform based on ITI
        if t == T:
            nIter = 1
        else:
            nIter = min(opts['maxIter'], round(Dist[t - 1, t]))

        # Calculate (unnormalized) posterior, !!not including reward!!
        if t == 1:
            N = np.zeros((1, opts['K']))
              
        elif t == 2:
            N = Z[0, :]
                
        else:
            # Sum the columns of the Z matrix.
            N = np.sum(Z[:t - 1, :], axis=0)
            
        if opts['c_alpha'][t - 1] == 0:
            prior = np.zeros((1, opts['K']))
            prior[0] = 1
        
        else:
            if t == 1:
                prior = np.ravel(np.zeros((1, opts['K'])))
            else:
                prior = S[:t - 1, t - 1] @ Z[:t - 1, :]

            if len(np.where(prior == 0)[0]) > 0:
                prior[np.where(prior == 0)[0][0]] = opts['c_alpha'][t - 1]

        L = prior / np.sum(prior)

        if t == 1:
            xsum = np.zeros((D, 1)) @ np.zeros((1, opts['K']))
            
        else:
            # Group as pairs of vectors. // Fortran order.
            X_pairs = X[0:t - 1, :].flatten('F')

            X_pairs_re = np.reshape(X_pairs, len(X_pairs))

            X_pairs_re_split = np.array(np.split(X_pairs_re, len(X_pairs_re) / D))

            X_pairs_re_split_transposed = X_pairs_re_split.transpose()

            xsum = X_pairs_re_split_transposed @ Z[0:t - 1, :]
        
        nu = opts['sx'] / (N + opts['sx']) + opts['sx']
        
        for d in range(D):
            xhat = xsum[d, :] / (N + opts['sx'])
            
            L = (L * norm.pdf(X[t - 1, d], xhat, np.sqrt(nu))).flatten()
              
        post = L / np.sum(L)
        
        V[t - 1] = np.dot(X[t - 1, :], W) @ post

        w_before_r = pd.concat([pd.DataFrame(w_before_r), pd.DataFrame(W)])

        post_before_r = pd.concat([pd.DataFrame(post_before_r), pd.DataFrame(post).T])

        if opts['nst'] == 1:
            V[t - 1] = 1 - norm.cdf(opts['theta'], V[t - 1], opts['lambda'])
                    
        for iter in range(1, nIter + 1):
            V_afterUS = X[t - 1, :] @ W
            post = L * norm.pdf(r[t - 1], V_afterUS, np.sqrt(opts['sr']))
            post = post / np.sum(post)
            rpe = (r[t - 1] - V_afterUS) * post
            rpe = np.tile(rpe, (D, 1))
            x = np.tile(X[t - 1, :], (opts['K'], 1)).transpose()
            W = W + opts['eta'][t - 1] * rpe * x

            if np.any(psi[t - 1] > 0):
                W = W * (1 - np.tile(post, (D, 1))) * psi[t - 1]

            t_vals.append(t)
            iter_vals.append(iter)

            # df operations.
            t_vals_w = pd.DataFrame(np.repeat(t_vals, len(W)), columns=['t'])
            t_vals_w.index += 1
            iter_vals_w = pd.DataFrame(np.repeat(iter_vals, len(W)), columns=['iter'])
            iter_vals_w.index += 1

            t_vals_post = pd.DataFrame(t_vals, columns=['t'])
            t_vals_post.index += 1
            iter_vals_post = pd.DataFrame(iter_vals, columns=['iter'])
            iter_vals_post.index += 1

            w_save = pd.concat([pd.DataFrame(w_save), pd.DataFrame(W)])
            w_save.index = np.arange(1, len(w_save) + 1)

            p_save = pd.concat([pd.DataFrame(p_save), pd.DataFrame(post).T])
            p_save.index = np.arange(1, len(p_save) + 1)

        zp_save = pd.concat([pd.DataFrame(zp_save), pd.DataFrame(post).T])
        zp_save.index = np.arange(1, len(zp_save) + 1)

        k = np.argmax(post)
        Z[t - 1, k] = 1

    w_save.columns = ['X' + str(i) for i in range(1, len(w_save.columns) + 1)]
    p_save.columns = ['X' + str(i) for i in range(1, len(p_save.columns) + 1)]
    zp_save.columns = np.arange(1, len(zp_save.columns) + 1)

    w_save = pd.concat([t_vals_w, iter_vals_w, w_save], axis=1)
    p_save = pd.concat([t_vals_post, iter_vals_post, p_save], axis=1)
    
    return {
        'opts': opts,
        'Dist': Dist,
        'V': V,
        'Z': Z,
        'S': S,
        'Zp': zp_save,
        'W': w_save,
        'P': p_save,
        'w': w_before_r,
        'p': post_before_r
    }

######################################################################
# Adapted from https://github.com/sjgershm/LCM/blob/master/LCM_lik.m #
######################################################################

def LCM_like(param, data, opts):
    if opts == None:
        opts = {}

    opts['alpha'] = param[0]

    ncol_data = data.shape[1]

    results = LCM_infer(data.iloc[:, 2:ncol_data].to_numpy(), opts)

    # Fit model results to CR values.
    # N is not needed, including it for clarity.
    N = len(results['V'])

    X = results['V']

    matrix_1 = X.T
    
    matrix_2 = data['CR'].to_numpy().reshape(-1, 1)

    matrix_3 = X

    b = (matrix_1 @ matrix_2) / (matrix_1 @ matrix_3)

    CRpred = X * b

    CRvals = data['CR'].to_numpy().reshape(-1, 1)

    sd = np.sqrt(np.mean((CRvals - CRpred) ** 2))

    # sum the log of the pdf.
    like = np.sum(norm.logpdf(CRvals - CRpred, 0, sd))

    latents = {}
    latents['latent_results'] = results
    latents['latent_b'] = b
    latents['latent_sd'] = sd
    latents['latent_CR'] = CRpred

    return {'likelihood': like, 'latents': latents}

def LCM_like_neg(param, data, opts):
    if opts == None:
        opts = {}
    
    results = LCM_like(param, data, opts)
    
    return -results['likelihood']

##########################################################################
# Adapted from https://github.com/ykunisato/lcmr/blob/master/R/fit_lcm.R #
# the code is from a dedicated R package for latent cause models.        #
##########################################################################

# Fit the LCM to the data:
# method 1: scipy's minimize with L-BFGS-B
# method 2: post mean
# method 3: scipy's minimize with Nelder-Mead

def LCM_fit(data, opts, param_range, est_method):
    if opts == None:
        opts = {}
    
    if est_method == None:
        est_method = 0
    
    default_param_range = {
            'a_L': 1e-31,
            'a_U': 10,
            'e_L': 1e-31,
            'e_U': 1
        }
    
    if param_range == None:
        param_range = default_param_range
    
    else: 
        for key in default_param_range.keys():
            if key not in param_range:
                param_range[key] = default_param_range[key]

    if est_method == 0:
        subject_ID = np.unique(data['Subject_ID'])

        fit = data.groupby('Subject_ID').apply(lambda x: estimate_with_minim(x, opts, param_range))

        fit = pd.DataFrame({'Subject_ID': subject_ID, 'alpha': fit['alpha'], 'nll': fit['nll']}).reset_index(drop=True)

    elif est_method == 1:
        print('Estimating subject alphas using post mean')
        alpha = np.linspace(param_range['a_L'], param_range['a_U'], 100)
        subject_ID = np.unique(data['Subject_ID'])

        fit = data.groupby('Subject_ID').apply(lambda x: estimate_with_post_mean(x, opts, alpha))
        fit = pd.DataFrame({'Subject_ID': subject_ID, 'alpha': fit['alpha'], 'logBF': fit['logBF']}).reset_index(drop=True)

    elif est_method == 2:
        subject_ID = np.unique(data['Subject_ID'])

        fit = data.groupby('Subject_ID').apply(lambda x: estimate_with_minim_NM(x, opts, param_range))
        fit = pd.DataFrame({'Subject_ID': subject_ID, 'alpha': fit['alpha'], 'nll': fit['nll']}).reset_index(drop=True)

    b = []
    sd = []

    for i in range(len(fit['Subject_ID'])):
        if pd.isna(fit['alpha'][i]):
            b[i] = np.nan
            sd[i] = np.nan
        else:
            estimate = LCM_like([fit['alpha'][i]], data, opts)
            b.append(estimate['latents']['latent_b'])
            sd.append(estimate['latents']['latent_sd'])

            b_vals = np.array(b).flatten()

    fit['b'] = b_vals
    fit['sd'] = sd

    return fit

#####################################################################################
# List of scipy optimizers https://docs.scipy.org/doc/scipy/reference/optimize.html #
#####################################################################################

# Method: 1    
def estimate_with_minim(data, opts, param_range):
    smallest_nll = np.inf
    param = []

    print('Estimating subject alphas using scipy minimize L-BFGS-B')

    for i in range(1, 201):
        init_param = np.random.uniform(param_range['a_L'], param_range['a_U'])

        try:
            results = minimize(LCM_like_neg, init_param, args=(data, opts), method='L-BFGS-B', bounds=[(param_range['a_L'], param_range['a_U'])])

            print(f"{i}  negative log likelihood: {results['fun']}  parameter: {results['x']}")
            
            if results['fun'] < smallest_nll:
                smallest_nll = results['fun']
                param = results['x']

        except:
            print(f"{i} Error in optimization using minimize")

        if len(param) != 0 and i in [10, 50, 75, 100, 125, 150, 175, 200]:
            break

    if len(param) == 0:
        param = np.nan

    df = pd.DataFrame({'alpha': [param[0]], 'nll': [smallest_nll]})

    return df

# Method: 2
def estimate_with_minim_NM(data, opts, param_range):
    smallest_nll = np.inf
    param = []

    print('Estimating subject alphas using scipy minimize Nelder-Mead')

    for i in range(1, 201):
        init_param = np.random.uniform(param_range['a_L'], param_range['a_U'])

        try:
            results = minimize(LCM_like_neg, init_param, args=(data, opts), method='Nelder-Mead', bounds=[(param_range['a_L'], param_range['a_U'])])

            print(f"{i}  negative log likelihood: {results['fun']}  parameter: {results['x']}")

            if results['fun'] < smallest_nll:
                smallest_nll = results['fun']
                param = results['x']

        except:
            print(f"{i} Error in optimization using minimize")

        if len(param) != 0 and i in [10, 50, 75, 100, 125, 150, 175, 200]:
            break

    if len(param) == 0:
        param = np.nan

    df = pd.DataFrame({'alpha': [param[0]], 'nll': [smallest_nll]})

    return df

# Method: 3
def estimate_with_post_mean(data, opts, alpha):
    
    likes = np.zeros(len(alpha))
    for i in range(len(alpha)):
        results = LCM_like([alpha[i]], data, opts)
        likes[i] = results['likelihood']
        
    L = np.log(np.sum(np.exp(likes)))
    P = np.exp(likes - L)

    post_mean_alpha = np.dot(alpha, P)
    
    logBF = L - np.log(len(alpha)) - likes[0]

    df = pd.DataFrame({'alpha': [post_mean_alpha], 'logBF': [logBF]})
    
    print(f"Subject {data['Subject_ID'].iloc[0]} estimated alpha: {post_mean_alpha} logBF: {logBF}")

    return df

##################
# Generate data #
##################

# def generate_stimuli_latent(trial_length=30):
#     US = np.concatenate([np.ones(10), np.zeros(20)])
#     CS = np.ones(trial_length)
#     Context = np.concatenate([np.ones(10), np.zeros(10), np.ones(10)])

#     data = np.column_stack((US, CS, Context))
#     return data

#############################################################
# Create Gymnasium environment for LCM_infer & LCM_infer_rw #
#############################################################
