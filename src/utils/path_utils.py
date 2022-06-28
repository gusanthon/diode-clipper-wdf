from pathlib import Path

utils_dir = Path(__file__).resolve().parent

root_dir = utils_dir.parent.parent

data_dir = root_dir / "data"
audio_dir = root_dir / "audio"
images_dir = root_dir / "images"
src_dir = root_dir / "src"