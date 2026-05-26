# ZaloAI Lyric Alignment

Word-level audio-to-lyrics alignment for Vietnamese songs, built for the **ZaloAI 2022** challenge.

## Overview

Given a song's audio and its lyrics, the system outputs each word annotated with a start and end timestamp (milliseconds).

**Pipeline:**
1. **ASR** — fine-tuned [Wav2Vec2](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) generates a log-probability matrix over tokens.
2. **Forced alignment** — CTC trellis + Viterbi backtracking maps tokens to time frames.
3. **Segmentation** — flat word alignments are grouped into sentences using ground-truth boundaries.

## Project Structure

```
ZaloAI_LyricAlignment/
├── config.py            # Shared constants and path configuration
├── predict_utils.py     # Alignment pipeline (ASR → CTC → word timestamps)
├── segment_utils.py     # Sentence-level segmentation and output formatting
├── predict.ipynb        # End-to-end inference notebook
├── requirements.txt
└── train/
    ├── Music_cutter.py  # Preprocessing: split songs into sentence-level clips
    └── Train.ipynb      # Fine-tuning Wav2Vec2 on Vietnamese lyrics data
```

## Setup

```bash
pip install -r requirements.txt
# kenlm (required for beam-search decoding) must be built from source:
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Configuration

Paths default to `/public_test/songs` and `/public_test/lyrics`. Override via environment variables without touching source code:

```bash
export SONGS_PATH=/path/to/songs
export LYRICS_PATH=/path/to/lyrics
```

## Usage

### Inference

Open and run `predict.ipynb`. It loads the fine-tuned checkpoint, iterates over the test set, and writes per-song JSON files to `submission/`.

Output format per song:
```json
[
  {
    "s": 1200,
    "e": 3400,
    "l": [
      {"d": "lần", "s": 1200, "e": 1650},
      {"d": "đầu", "s": 1700, "e": 2100}
    ]
  }
]
```

### Training

1. **Preprocess data** — split full songs into sentence-level clips:
   ```bash
   cd train
   python Music_cutter.py
   ```
2. **Fine-tune** — open and run `train/Train.ipynb`.

## Models

| Model | Source |
|---|---|
| Pre-trained ASR | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` |
| Fine-tuned checkpoint | `hungngocphat01/Checkpoint_zaloAI_11_19_2022` |
| 4-gram language model | `vi_lm_4grams.bin` (local) |
