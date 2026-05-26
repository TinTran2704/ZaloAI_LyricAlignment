"""
End-to-end lyric alignment: given audio and a lyrics string, produce word-level timestamps.
"""

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from config import BLANK_ID, LYRICS_PATH, SAMPLE_RATE, SONGS_PATH

logger = logging.getLogger(__name__)


def end_to_end_align(
    song: Dict[str, Any],
    processor: Any,
    model: Any,
) -> List[Dict[str, Any]]:
    """
    Run the full alignment pipeline for a single song.

    Parameters
    ----------
    song : dict
        Must contain ``array`` (torch.Tensor) and ``text`` (str).
    processor : Wav2Vec2Processor
    model : Wav2Vec2ForCTC

    Returns
    -------
    list of dicts ``{"d": word, "s": start_ms, "e": end_ms}``
    """
    input_values = processor(
        song["array"],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits[0]
        asr_output = torch.log_softmax(logits, dim=-1)

    return alignment_from_asr(song["array"], asr_output, song["text"], processor)


def alignment_from_asr(
    waveform: torch.Tensor,
    asr_output: torch.Tensor,
    transcript: str,
    processor: Any,
) -> List[Dict[str, Any]]:
    """
    Compute word-level alignment from an ASR log-probability matrix.

    Returns
    -------
    list of dicts ``{"d": word, "s": start_ms, "e": end_ms}``
    """
    with processor.as_target_processor():
        tokens = processor(transcript).input_ids

    trellis = get_trellis(asr_output, tokens)
    path = backtrack(trellis, asr_output, tokens)
    segments = merge_repeats(path, transcript)
    word_segments = merge_words(segments)

    aligned = []
    for i, word_seg in enumerate(word_segments):
        start_s, end_s = get_word_seconds(i, waveform, trellis, word_segments)
        aligned.append({
            "d": word_seg.label,
            "s": int(start_s * 1000),
            "e": int(end_s * 1000),
        })

    return aligned


def load_single_example(file_id: str) -> Dict[str, Any]:
    """
    Load audio and lyrics for one example and normalise the lyrics string.

    Parameters
    ----------
    file_id : str
        Filename without extension, e.g. ``"song_001"``.

    Returns
    -------
    dict with keys ``filename``, ``array``, ``text``
    """
    audio_arr = read_audio(os.path.join(SONGS_PATH, file_id + ".wav"))

    with open(os.path.join(LYRICS_PATH, file_id + ".json"), encoding="utf-8") as f:
        lyrics_data = json.load(f)

    raw = " ".join(tok["d"].lower() for sent in lyrics_data for tok in sent["l"])
    normalised = unicodedata.normalize("NFC", "".join(c for c in raw if c.isalnum() or c == " "))

    return {"filename": file_id, "array": audio_arr, "text": normalised}


def read_audio(filename: str) -> torch.Tensor:
    """Read a WAV file, convert to mono and resample to ``SAMPLE_RATE``."""
    speech, sr = sf.read(filename)

    if speech.ndim == 2:
        speech = np.mean(speech, axis=1)

    speech = librosa.resample(y=speech, orig_sr=sr, target_sr=SAMPLE_RATE)
    return torch.tensor(speech, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CTC forced-alignment (adapted from PyTorch Audio tutorial)
# https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
# ---------------------------------------------------------------------------

def get_trellis(
    emission: torch.Tensor,
    tokens: List[int],
    blank_id: int = BLANK_ID,
) -> torch.Tensor:
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Extra dimensions: one for the <SoS> token, one for the time axis.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],   # stay at same token
            trellis[t, :-1] + emission[t, tokens],    # advance to next token
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: List[int],
    blank_id: int = BLANK_ID,
) -> List[Point]:
    # j / t are trellis indices; emission indices are offset by -1.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path: List[Point] = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(Point(j - 1, t - 1, prob))

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align: transcript and audio may be mismatched")

    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self) -> str:
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self) -> int:
        return self.end - self.start


def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    i1, i2 = 0, 0
    segments: List[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(
            transcript[path[i1].token_index],
            path[i1].time_index,
            path[i2 - 1].time_index + 1,
            score,
        ))
        i1 = i2
    return segments


def merge_words(segments: List[Segment], separator: str = " ") -> List[Segment]:
    words: List[Segment] = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join(seg.label for seg in segs)
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def get_word_seconds(
    i: int,
    waveform: torch.Tensor,
    trellis: torch.Tensor,
    word_segments: List[Segment],
) -> Tuple[float, float]:
    ratio = waveform.unsqueeze(0).size(1) / (trellis.size(0) - 1)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    return x0 / SAMPLE_RATE, x1 / SAMPLE_RATE
