from pathlib import Path

utils_dir = Path(__file__).resolve().parent

root_dir = utils_dir.parent.parent

data_path = root_dir / "data"
audio_path = root_dir / "audio"
images_path = root_dir / "images"
src_path = root_dir / "src"