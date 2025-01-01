import pytest
from parallelopedia.http.server import RangedRequest, InvalidRangeRequest, RangeRequestTooLarge

def test_ranged_request_valid_range():
    rr = RangedRequest('bytes=0-499')
    rr.set_file_size(1000)
    assert rr.first_byte == 0
    assert rr.last_byte == 499
    assert rr.num_bytes_to_send == 500

def test_ranged_request_suffix_length():
    rr = RangedRequest('bytes=-500')
    rr.set_file_size(1000)
    assert rr.first_byte == 500
    assert rr.last_byte == 999
    assert rr.num_bytes_to_send == 500

def test_ranged_request_invalid_range():
    with pytest.raises(InvalidRangeRequest):
        RangedRequest('bytes=1000-500')

def test_ranged_request_too_large():
    rr = RangedRequest('bytes=0-2147483648')  # 2GB + 1 byte
    with pytest.raises(RangeRequestTooLarge):
        rr.set_file_size(2147483649)  # 2GB + 1 byte

def test_ranged_request_no_end():
    rr = RangedRequest('bytes=500-')
    rr.set_file_size(1000)
    assert rr.first_byte == 500
    assert rr.last_byte == 999
    assert rr.num_bytes_to_send == 500
