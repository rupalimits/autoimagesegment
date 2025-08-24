import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from image_captioning.config.core import ROOT, config

with open(ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()