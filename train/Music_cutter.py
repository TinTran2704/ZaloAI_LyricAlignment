"""
Splits full-song audio and label files into sentence-level segments for training.

Usage
-----
Run from the ``train/`` directory:
    python Music_cutter.py

Expected layout::

    train/
    ├── labels/      # ground-truth JSON label files
    ├── songs/       # full-song WAV files
    ├── new-labels/  # output: sentence-level JSON labels  (created beforehand)
    └── new-songs/   # output: sentence-level WAV clips    (created beforehand)
"""

import json
import logging
import os

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_PATH = "labels"
SONGS_PATH = "songs"
NEW_LABEL_PATH = "new-labels"
NEW_SONGS_PATH = "new-songs"


def get_audio_segment(
    audiodata: np.ndarray,
    samplerate: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    """Extract the audio slice between ``start_ms`` and ``end_ms`` milliseconds."""
    start = int(start_ms / 1000 * samplerate)
    end = int(end_ms / 1000 * samplerate)
    return audiodata[start:end]


def main() -> None:
    file_ids = [f.split(".")[0] for f in os.listdir(LABEL_PATH) if f]

    for file_id in file_ids:
        with open(os.path.join(LABEL_PATH, file_id + ".json"), encoding="utf-8") as f:
            label_data = json.load(f)

        audiodata, samplerate = sf.read(os.path.join(SONGS_PATH, file_id + ".wav"))
        if audiodata.ndim == 2:
            audiodata = np.mean(audiodata, axis=1)

        for i, sent_label in enumerate(label_data):
            orig_start = sent_label["s"]
            new_label = {
                "orig_s": orig_start,
                "l": [
                    {
                        "s": tok["s"] - orig_start,
                        "e": tok["e"] - orig_start,
                        "d": tok["d"].lower(),
                    }
                    for tok in sent_label["l"]
                ],
            }

            with open(os.path.join(NEW_LABEL_PATH, f"{file_id}-{i}.json"), "w", encoding="utf-8") as f:
                json.dump(new_label, f)

            sent_audio = get_audio_segment(audiodata, samplerate, sent_label["s"], sent_label["e"])
            sf.write(os.path.join(NEW_SONGS_PATH, f"{file_id}-{i}.wav"), sent_audio, samplerate)

        logger.info("Processed %s (%d sentences)", file_id, len(label_data))


if __name__ == "__main__":
    main()
