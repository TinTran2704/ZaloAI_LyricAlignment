"""
Post-processing: group flat word-level alignments into sentence-level segments.
"""

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Set

from config import LYRICS_PATH

logger = logging.getLogger(__name__)


def do_segmentation_transform(
    pred_labels: List[Dict[str, Any]],
    annotate_lyrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full segmentation pipeline: detect noisy songs, segment, and format output.

    Parameters
    ----------
    pred_labels : list
        Word-level alignments from ``predict_utils.end_to_end_align``.
    annotate_lyrics : dict
        Ground-truth labels from ``load_annotate_lyrics``.

    Returns
    -------
    dict mapping file_id -> ZaloAI submission format
    """
    noise_songs = _get_noise_songs(annotate_lyrics, pred_labels)
    segmented_labels = do_segmentation(pred_labels, annotate_lyrics, noise_songs)
    return _transform_result(segmented_labels)


def load_annotate_lyrics(file_ids: List[str]) -> Dict[str, Any]:
    """
    Load ground-truth label JSON files for the given file IDs.

    Parameters
    ----------
    file_ids : list[str]
        Filenames without extension.

    Returns
    -------
    dict mapping file_id -> parsed JSON
    """
    annotate_lyrics: Dict[str, Any] = {}
    for file_id in file_ids:
        if file_id == ".ipynb_checkpoints":
            continue
        with open(os.path.join(LYRICS_PATH, file_id + ".json"), encoding="utf-8") as f:
            annotate_lyrics[file_id] = json.load(f)
    return annotate_lyrics


def _count_len_sent_annotate(song: List[Dict[str, Any]]) -> List[int]:
    """Return the token count for each sentence in ``song``."""
    return [len(sent["l"]) for sent in song]


def _get_noise_songs(
    annotate_lyrics: Dict[str, Any],
    pred_labels: List[Dict[str, Any]],
) -> Set[str]:
    """
    Return song IDs where predicted token count differs from the annotation.
    These songs are skipped during segmentation because sentence boundaries
    cannot be reliably matched.
    """
    num_tok_annotate = {
        song_id: sum(_count_len_sent_annotate(song))
        for song_id, song in annotate_lyrics.items()
    }
    num_tok_pred = {song["song"]: len(song["alignment"]) for song in pred_labels}

    assert set(num_tok_annotate.keys()) == set(num_tok_pred.keys()), (
        "Mismatch between annotated and predicted song IDs"
    )

    noise_songs = {
        song_id
        for song_id in num_tok_annotate
        if num_tok_annotate[song_id] != num_tok_pred[song_id]
    }
    if noise_songs:
        logger.warning("Skipping %d noisy song(s): %s", len(noise_songs), noise_songs)
    return noise_songs


def do_segmentation(
    pred_labels: List[Dict[str, Any]],
    annotate_lyrics: Dict[str, Any],
    noise_songs: Set[str],
) -> Dict[str, Any]:
    """
    Split flat word alignments into per-sentence groups using annotated boundaries.
    Songs in ``noise_songs`` are skipped.
    """
    new_labels: Dict[str, Any] = {}

    for song in pred_labels:
        song_id = song["song"]
        if song_id in noise_songs:
            continue

        alignment = deepcopy(song["alignment"])
        tok_segs = _count_len_sent_annotate(annotate_lyrics[song_id])

        segs = []
        for n in tok_segs:
            segs.append(alignment[:n])
            alignment = alignment[n:]

        new_labels[song_id] = segs

    return new_labels


def _transform_result(segmented_labels: Dict[str, Any]) -> Dict[str, Any]:
    """Format segmented alignment into the ZaloAI submission schema."""
    result: Dict[str, Any] = {}
    for file_id, song in segmented_labels.items():
        try:
            result[file_id] = [
                {"s": sent[0]["s"], "e": sent[-1]["e"], "l": sent}
                for sent in song
            ]
        except (IndexError, KeyError) as exc:
            logger.warning("Skipping %s: failed to format result (%s)", file_id, exc)

    return result
