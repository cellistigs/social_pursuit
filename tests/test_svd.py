## Test different svd implementations. 
import pytest
from social_pursuit.labeled import LabeledData
from social_pursuit.onlineRPCA.rpca.pcp import pcp,pcp_jax 
import numpy as np
import os

here = os.path.abspath(os.path.dirname(__file__))
training_dir = os.path.join(here,"test_fixtures/training_data/")

labeled_data = os.path.join(training_dir,"CollectedData_Taiga.h5")
additionalpath = training_dir

@pytest.fixture
def get_dataarray():
    data = LabeledData(labeled_data,additionalpath)
    frameinds = [1,2,3,4]
    frames = data.get_images(frameinds)
    flatframes = [f.flatten() for f in frames]
    dataarray = np.stack(flatframes,axis = -1)
    yield dataarray

def test_base_svd(get_dataarray):
    pcp(get_dataarray)
#%def test_jax_svd(get_dataarray):
#%    pcp_jax(get_dataarray).block_until_ready()
#%    assert 0 


