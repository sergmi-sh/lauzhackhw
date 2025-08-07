from pathlib import Path
import os
from typing import Any, override

import torchaudio
import pandas as pd

from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):
    def __init__(self, flac_dir_path: str, protocol_path: str, *args, **kwargs):
        flac_dir = Path(flac_dir_path)
        protocol_file = Path(protocol_path)
        self.flac_dir = flac_dir / "flac"
        # Columns: speaker filename system blank label
        self.protocol = pd.read_csv(protocol_file, sep=" ", header=None)
        index: list[dict[str, Any]] = []
        for idx in range(len(self.protocol)):
            file_id: str = self.protocol.iloc[idx, 1]
            filename: str = file_id + ".flac"
            label = 1 if self.protocol.iloc[idx, 4] == "bonafide" else 0
            flac_path = self.flac_dir / filename
            index.append({"path": flac_path, "label": label, "file_id": file_id})
        super().__init__(index, *args, **kwargs)

    @override
    def load_object(self, path):
        audio, _ = torchaudio.load(path)
        return audio
