import pytest
from camex_faceid.io_utils import FileClock, claim_directory


def test_offset_clock():
    c = FileClock(10)
    assert [c.timestamp(i, 1000 + i * 100) for i in range(3)] == [0, 100, 200]


def test_stuck_clock():
    c = FileClock(10)
    c.timestamp(0, 1000)
    with pytest.raises(ValueError):
        c.timestamp(1, 1000)


def test_unrelated_files_protected(tmp_path):
    (tmp_path / "demo.mp4").write_bytes(b"source")
    with pytest.raises(FileExistsError):
        claim_directory(tmp_path, "abc")
    assert (tmp_path / "demo.mp4").read_bytes() == b"source"


def test_input_inside_output_rejected(tmp_path):
    with pytest.raises(ValueError):
        claim_directory(tmp_path, "abc", [tmp_path / "video.mp4"])


def test_only_own_partial_run_reused(tmp_path):
    d = tmp_path / "job"
    claim_directory(d, "abc")
    claim_directory(d, "abc")
    with pytest.raises(FileExistsError):
        claim_directory(d, "different")


def test_concurrent_writer_rejected(tmp_path):
    d = tmp_path / "job"
    lock = claim_directory(d, "abc")
    try:
        with pytest.raises(RuntimeError):
            claim_directory(d, "abc")
    finally:
        lock.close()
    claim_directory(d, "abc").close()
