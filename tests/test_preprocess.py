from PIL import Image

from orthovision.preprocess.transform import (
    PreprocessConfig,
    normalize_image,
    resize_pad,
    window_intensity,
)

CFG = PreprocessConfig(p_low=0.5, p_high=99.5, target=224, keep_aspect_ratio=True, pad=True)


def test_normalize_returns_rgb_square(tmp_path):
    p = tmp_path / "x.png"
    Image.new("L", (800, 400), 128).save(p)
    out = normalize_image(p, CFG)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_resize_pad_preserves_aspect_and_pads(tmp_path):
    img = Image.new("RGB", (800, 400), (200, 200, 200))
    out = resize_pad(img, 224, keep_aspect=True, pad=True)
    assert out.size == (224, 224)
    # wide image -> vertical padding: top and bottom rows are the pad color (black)
    assert out.getpixel((112, 0)) == (0, 0, 0)


def test_window_intensity_expands_contrast():
    # values confined to a narrow band should be stretched toward the full range
    img = Image.new("L", (10, 10))
    img.putdata([100] * 50 + [150] * 50)
    out = window_intensity(img, 0.5, 99.5)
    vals = set(out.getdata())
    assert min(vals) == 0 and max(vals) == 255
