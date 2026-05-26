"""
Shared constants and path configuration.
Override paths via environment variables to avoid editing source code.
"""

import os

# Wav2Vec2 Vietnamese 250h model: index of the blank/pad token in its vocabulary
BLANK_ID = 109

# All Wav2Vec2 models expect 16 kHz input
SAMPLE_RATE = 16_000

SONGS_PATH = os.getenv("SONGS_PATH", "/public_test/songs")
LYRICS_PATH = os.getenv("LYRICS_PATH", "/public_test/lyrics")
