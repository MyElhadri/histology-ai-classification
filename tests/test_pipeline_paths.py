"""Tests to verify that pipeline.py correctly resolves image paths.

Specifically, ensures that an image_path like:
    data/raw/nuinsseg_human_22_original/class/image.png
is NOT transformed into:
    data/raw/.../data/raw/.../image.png
"""

import sys
import types
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# TensorFlow is not installed locally. We stub it before importing pipeline
# so that resolve_image_path (a pure-Python function) can be tested.
# ---------------------------------------------------------------------------
_tf_stub = types.ModuleType("tensorflow")
_tf_stub.Tensor = object
_tf_stub.data = types.SimpleNamespace(AUTOTUNE=-1, Dataset=object)
_tf_stub.keras = types.SimpleNamespace(
    utils=types.SimpleNamespace(to_categorical=lambda *a, **k: None),
    applications=types.SimpleNamespace(
        densenet=types.SimpleNamespace(preprocess_input=lambda x: x)
    ),
)
_tf_stub.image = types.SimpleNamespace(
    decode_image=None,
    resize=None,
    random_flip_left_right=None,
    random_flip_up_down=None,
    random_brightness=None,
)
_tf_stub.io = types.SimpleNamespace(read_file=None)
_tf_stub.cast = None
sys.modules.setdefault("tensorflow", _tf_stub)

from src.data.pipeline import resolve_image_path  # noqa: E402


def test_resolve_relative_path(tmp_path: Path) -> None:
    """A relative path is correctly joined with project_root."""
    # Create a fake image file
    fake_dir = tmp_path / "data" / "raw" / "class_a"
    fake_dir.mkdir(parents=True)
    fake_img = fake_dir / "image_001.png"
    fake_img.write_bytes(b"fake")

    rel_path = "data/raw/class_a/image_001.png"
    resolved = resolve_image_path(rel_path, project_root=tmp_path)

    assert resolved == fake_img.resolve()


def test_resolve_no_duplicate_prefix(tmp_path: Path) -> None:
    """Verifies the path does not contain a duplicated dataset_root prefix."""
    fake_dir = tmp_path / "data" / "raw" / "nuinsseg_human_22_original" / "brain"
    fake_dir.mkdir(parents=True)
    fake_img = fake_dir / "brain_01.png"
    fake_img.write_bytes(b"fake")

    rel_path = "data/raw/nuinsseg_human_22_original/brain/brain_01.png"
    resolved = resolve_image_path(rel_path, project_root=tmp_path)

    # The resolved path should contain exactly ONE occurrence of 'nuinsseg_human_22_original'
    parts_str = str(resolved)
    count = parts_str.count("nuinsseg_human_22_original")
    assert count == 1, (
        f"Expected exactly 1 occurrence of 'nuinsseg_human_22_original' in path, "
        f"got {count}: {parts_str}"
    )


def test_resolve_absolute_path(tmp_path: Path) -> None:
    """An absolute path is used directly without prepending project_root."""
    fake_dir = tmp_path / "images"
    fake_dir.mkdir()
    fake_img = fake_dir / "abs_image.png"
    fake_img.write_bytes(b"fake")

    abs_path = str(fake_img.resolve())
    resolved = resolve_image_path(abs_path, project_root=Path("/some/other/root"))

    assert resolved == fake_img.resolve()


def test_resolve_missing_file_raises(tmp_path: Path) -> None:
    """A FileNotFoundError is raised for a non-existent image."""
    with pytest.raises(FileNotFoundError):
        resolve_image_path("data/raw/nonexistent/image.png", project_root=tmp_path)
