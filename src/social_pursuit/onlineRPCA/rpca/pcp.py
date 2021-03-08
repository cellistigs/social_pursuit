import numpy as np
from tqdm import tqdm as tqdm
from numba import jit
import jax
import jax.numpy as jnp
from .utility import thres

def pcp(M, lam=np.nan, mu=np.nan, factor=1, tol=10**(-7), maxit=1000):
    """ Robust PCA (Candes et al., 2009)
    
    This code solves the Principal Component Pursuit
    min_M { ||L||_* + lam*||S(:)||_1 }
    s.t. M = S+L
    using an Augmented Lagrange Multiplier (ALM) algorithm.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the PCP algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter used in augmented Lagrangian which is in front of 
    the ||M-L-S||_F^2 term (default NaN). When mu is set to NaN, the value 0.25/np.abs(M).mean() 
    will be used in the PCP algorithm.
    
    factor: tuning parameter (default 1). When lam is set to NaN,  lam will take the value 1/sqrt(max(m, n)) * factor
    in the PCP algorithm, where M is a m by n matrix. When lam is not NaN, factor is ignored.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 1000).
    
    Returns
    ----------
    L : array-like, low-rank matrix.
    
    S : array-like, sparse matrix.
    
    niter : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Candes, Emmanuel J., et al. Robust principal component analysis. 
        Journal of the ACM (JACM) 58.3 (2011): 11.
    
    Yuan, Xiaoming, and Junfeng Yang. Sparse and low-rank matrix decomposition via alternating direction methods. 
        preprint 12 (2009). [tuning method]
    
    """            
    # initialization
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable
 
    # parameter setting
    if np.isnan(mu):
        mu = 0.25/np.abs(M).mean()
    if np.isnan(lam):
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    # main
    for niter in range(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        # L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # L has closed form solution (singular value thresholding)
        Y = X - S;
        dL = L;       
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);
        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL
        
        # S - subproblem 
        # S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # Define element wise softshinkage operator as 
        #     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        # S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        Y = X - L
        dS = S
        S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z;
        
        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol: 
            break
    
    return L, S, niter, rank

@jit(nopython=True)
def pcp_numba(M, lam=np.nan, mu=np.nan, factor=1, tol=10**(-7), maxit=1000):
    """ Robust PCA (Candes et al., 2009)
    
    This code solves the Principal Component Pursuit
    min_M { ||L||_* + lam*||S(:)||_1 }
    s.t. M = S+L
    using an Augmented Lagrange Multiplier (ALM) algorithm.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the PCP algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter used in augmented Lagrangian which is in front of 
    the ||M-L-S||_F^2 term (default NaN). When mu is set to NaN, the value 0.25/np.abs(M).mean() 
    will be used in the PCP algorithm.
    
    factor: tuning parameter (default 1). When lam is set to NaN,  lam will take the value 1/sqrt(max(m, n)) * factor
    in the PCP algorithm, where M is a m by n matrix. When lam is not NaN, factor is ignored.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 1000).
    
    Returns
    ----------
    L : array-like, low-rank matrix.
    
    S : array-like, sparse matrix.
    
    niter : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Candes, Emmanuel J., et al. Robust principal component analysis. 
        Journal of the ACM (JACM) 58.3 (2011): 11.
    
    Yuan, Xiaoming, and Junfeng Yang. Sparse and low-rank matrix decomposition via alternating direction methods. 
        preprint 12 (2009). [tuning method]
    
    """            
    # initialization
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable
 
    # parameter setting
    if np.isnan(mu):
        mu = 0.25/np.abs(M).mean()
    if np.isnan(lam):
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    # main
    for niter in range(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        # L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # L has closed form solution (singular value thresholding)
        Y = X - S;
        dL = L;       
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False);
        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL
        
        # S - subproblem 
        # S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # Define element wise softshinkage operator as 
        #     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        # S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        Y = X - L
        dS = S

        Y = np.maximum(Y - lam/mu, 0)
        S = Y + np.minimum(Y + lam/mu, 0)
        #S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z;
        
        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol: 
            break
    
    return L, S, niter, rank

def pcp_jax(M, lam=jnp.nan, mu=jnp.nan, factor=1, tol=10**(-7), maxit=1000):
    """ Robust PCA (Candes et al., 2009)
    
    This code solves the Principal Component Pursuit
    min_M { ||L||_* + lam*||S(:)||_1 }
    s.t. M = S+L
    using an Augmented Lagrange Multiplier (ALM) algorithm.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the PCP algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter used in augmented Lagrangian which is in front of 
    the ||M-L-S||_F^2 term (default NaN). When mu is set to NaN, the value 0.25/jnp.abs(M).mean() 
    will be used in the PCP algorithm.
    
    factor: tuning parameter (default 1). When lam is set to NaN,  lam will take the value 1/sqrt(max(m, n)) * factor
    in the PCP algorithm, where M is a m by n matrix. When lam is not NaN, factor is ignored.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 1000).
    
    Returns
    ----------
    L : array-like, low-rank matrix.
    
    S : array-like, sparse matrix.
    
    niter : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Candes, Emmanuel J., et al. Robust principal component analysis. 
        Journal of the ACM (JACM) 58.3 (2011): 11.
    
    Yuan, Xiaoming, and Junfeng Yang. Sparse and low-rank matrix decomposition via alternating direction methods. 
        preprint 12 (2009). [tuning method]
    
    """            
    # initialization
    m, n = M.shape
    unobserved = jnp.isnan(M)
    M[unobserved] = 0
    S = jnp.zeros((m,n))
    L = jnp.zeros((m,n))
    Lambda = jnp.zeros((m,n)) # the dual variable
 
    # parameter setting
    if jnp.isnan(mu):
        mu = 0.25/jnp.abs(M).mean()
    if jnp.isnan(lam):
        lam = 1/jnp.sqrt(max(m,n)) * float(factor)
        
    # main
    for niter in tqdm(range(maxit)):

        #L,S,Lambda,rank,RelChg = fastil(M,L,S,Lambda,mu,lam,unobserved)
        L,S,Lambda,rank,RelChg = inner_loop(M,L,S,Lambda,mu,lam,unobserved)

        #normLS = jnp.linalg.norm(jnp.concatenate((S,L), axis=1), 'fro')              
        ## dS, dL record the change of S and L, only used for stopping criterion

        #X = Lambda / mu + M
        ## L - subproblem
        ## L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        ## L has closed form solution (singular value thresholding)
        #Y = X - S;
        #dL = L;       
        #print(Y.shape)
        #U, sigmas, V = jnp.linalg.svd(Y, full_matrices=False);
        #VT, sigmas, UT = jnp.linalg.svd(Y.T, full_matrices=False);
        ##VT, sigmas, UT = jax.scipy.linalg.svd(Y, full_matrices=False,overwrite_a = True);
        #U,V = UT.T,VT.T
        #rank = (sigmas > 1/mu).sum()
        #Sigma = jnp.diag(sigmas[0:rank] - 1/mu)
        #L = jnp.dot(jnp.dot(U[:,0:rank], Sigma), V[0:rank,:])
        #dL = L - dL
        #
        ## S - subproblem 
        ## S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        ## Define element wise softshinkage operator as 
        ##     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        ## S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        #Y = X - L
        #dS = S
        #S = thres(Y, lam/mu) # softshinkage operator
        #dS = S - dS

        ## Update Lambda (dual variable)
        #Z = M - S - L
        #jax.ops.index_update(Z,unobserved,0)
        ##Z[unobserved] = 0
        #Lambda = Lambda + mu * Z;
        #
        ## stopping criterion
        #RelChg = jnp.linalg.norm(jnp.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        if RelChg < tol: 
            break
    
    return L, S, niter, rank

def inner_loop(M,L,S,Lambda,mu,lam,unobserved):
        normLS = jnp.linalg.norm(jnp.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        # L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # L has closed form solution (singular value thresholding)
        Y = X - S;
        dL = L;       
        U, sigmas, V = jnp.linalg.svd(Y, full_matrices=False);
        rank = (sigmas > 1/mu).sum()

        ## These must keep static shape to work with Jit. 
        #Sigma = jnp.diag(sigmas[0:rank] - 1/mu)
        #L = jnp.dot(jnp.dot(U[:,0:rank], Sigma), V[0:rank,:])
        #dL = L - dL
        ## Just take the top rank; the result should be the same. 
        sigmasprime = jax.ops.index_update(sigmas-1/mu,jax.ops.index[rank:],0)

        Sigma = jnp.diag(sigmasprime)
        L = jnp.dot(jnp.dot(U, Sigma), V)
        dL = L - dL

        
        # S - subproblem 
        # S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # Define element wise softshinkage operator as 
        #     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        # S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        Y = X - L
        dS = S
        S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Zprime = jax.ops.index_update(Z,unobserved,0)
        #Z[unobserved] = 0
        Lambda = Lambda + mu * Zprime;
        
        # stopping criterion
        RelChg = jnp.linalg.norm(jnp.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)

        return L,S,Lambda,rank,RelChg
