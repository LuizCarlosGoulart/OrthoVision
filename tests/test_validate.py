from orthovision.validate.rules import ImageRules, check_image
from orthovision.validate.report import build_report, validate_dir

RULES = ImageRules(
    min_width=512, min_height=256, allowed_formats=frozenset({"png", "jpg", "jpeg"})
)


def test_valid_image_passes(make_image):
    assert check_image(make_image("ok.png", size=(800, 400)), RULES) is None


def test_undersized_image_fails_on_resolution(make_image):
    assert check_image(make_image("small.png", size=(100, 100)), RULES) == "resolution"


def test_disallowed_format_fails(make_image):
    assert check_image(make_image("x.bmp", size=(800, 400), fmt="BMP"), RULES) == "format"


def test_corrupt_image_is_unreadable(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"not a real png")
    assert check_image(p, RULES) == "unreadable"


def test_report_counts_by_reason(tmp_path, make_image):
    make_image("store/ok1.png", size=(800, 400))
    make_image("store/ok2.png", size=(800, 400))
    make_image("store/small.png", size=(50, 50))
    passed, excluded = validate_dir(tmp_path / "store", RULES)
    report = build_report(passed, excluded)
    assert report == {
        "total": 3,
        "passed": 2,
        "excluded": 1,
        "excluded_by_reason": {"resolution": 1},
    }
