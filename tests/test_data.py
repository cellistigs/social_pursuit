# Test the data module
import pytest
#from botocore.stub import Stubber
from social_pursuit.data import PursuitTraces#PursuitVideo,PursuitTraces,s3_client,transfer_if_not_found


#@pytest.fixture(autouse=False)
#def s3_stub():
#    with Stubber(s3_client) as stubber:
#        yield stubber
#        stubber.assert_no_pending_responses()


def test_PursuitTraces():
    a = PursuitTraces("test_fixtures/trace_template.json")

#def test_transfer_if_not_found(s3_stub):
#    s3_stub.add_response('download_file',expected_params = {"bucketname":"froemkecarcealabs.behaviordata","objname":"RT_Cohousing/Trial21.mpg","filename":""},service_response = {})
#    filename = "s3://froemkecarcealabs.behaviordata/RT_Cohousing/Trial21.mpg"
#    transfer_if_not_found(filename)
#    assert 0
#
#def test_load_spec():
#    a = PursuitVideo("test_fixtures/template.json")
#
#def test_parse_videopaths():
#    a = PursuitVideo("test_fixtures/template.json")
#    a = PursuitVideo("test_fixtures/template_nosource.json")
#    with pytest.raises(AssertionError):
#        assert PursuitVideo("test_fixtures/template_source_misformat.json")

