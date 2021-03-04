import social_pursuit.onlineRPCA.rpca.pcp as pcp
import social_pursuit.onlineRPCA.rpca.spca as spca
import social_pursuit.onlineRPCA.rpca.spca2 as spca2
import social_pursuit.onlineRPCA.rpca.mwrpca as mwrpca
import social_pursuit.onlineRPCA.rpca.omwrpca as omwrpca
import pytest
import numpy as np

def test_pcp():
    dim = 20
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,niter,rank = pcp.pcp(data)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 1e-4
    assert np.linalg.norm(sparse-S, ord = "fro")< 1e-5

def test_spca():
    dim = 200
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,niter,rank = spca.spca(data)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 5e-1
    assert np.linalg.norm(sparse-S, ord = "fro")< 10 

def test_spca2():
    dim = 200
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,niter,rank = spca2.spca2(data)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 5e-1
    assert np.linalg.norm(sparse-S, ord = "fro")< 10 

def test_mwrpca():
    dim = 20
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,rank = mwrpca.mwrpca(data,burnin=20,win_size = 20)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 5e-2
    assert np.linalg.norm(sparse-S, ord = "fro")< 10 

def test_omwrpca():
    dim = 20
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,rank = omwrpca.omwrpca(data,burnin=20,win_size = 20)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 5e-1
    assert np.linalg.norm(sparse-S, ord = "fro")< 20 

def test_omwrpca():
    dim = 20
    lr_0 = np.random.randn(dim,)
    lr_1 = np.random.randn(dim,)
    lr = np.outer(lr_0,lr_1)

    out = 6 
    magnitude = 100
    sparseinds = np.random.choice(dim**2,out)
    sparse_init = np.zeros(dim**2)
    sparse_init[sparseinds] = np.random.randint(20,magnitude,out) 
    sparse = np.reshape(sparse_init,(dim,dim))

    data = lr+sparse
    L,S,rank = omwrpca.omwrpca(data,burnin=20,win_size = 20)
    assert rank == 1
    assert np.linalg.norm(lr-L, ord = "fro")/np.linalg.norm(lr,ord = "fro") < 5e-1
    assert np.linalg.norm(sparse-S, ord = "fro")< 20 
