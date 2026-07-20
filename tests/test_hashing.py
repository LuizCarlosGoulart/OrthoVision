from orthovision.hashing import sha256_file


def test_sha256_is_deterministic_and_content_addressed(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"orthovision")
    assert sha256_file(a) == sha256_file(a)

    b = tmp_path / "b.bin"
    b.write_bytes(b"orthovision")
    assert sha256_file(a) == sha256_file(b)  # same content -> same hash

    b.write_bytes(b"different")
    assert sha256_file(a) != sha256_file(b)
