import soundfile as sf
import numpy as np
import json
import os

files = os.listdir("labels")

file_ids = [f.split(".")[0] for f in files]
file_ids = filter(lambda x: len(x) > 0, file_ids)

LABEL_PATH = "labels"
SONGS_PATH = "songs"

NEW_LABEL_PATH = "new-labels"
NEW_SONGS_PATH = "new-songs"

def get_audio_segment(audiodata, samplerate, start_ms, end_ms):
    d = audiodata[int(start_ms / 1000*samplerate):int(end_ms / 1000*samplerate)]
    return d

for file_id in file_ids:
    # Read label file
    label_file = os.path.join(LABEL_PATH, file_id + ".json")
    label_data = json.load(open(label_file, encoding='utf-8'))
    
    # Read audio file
    sound_file = os.path.join(SONGS_PATH, file_id + ".wav")
    audiodata, samplerate = sf.read(sound_file)
    audiodata = np.mean(audiodata, axis=1)
    
    # Break segment into sentences
    for i, sent_label in enumerate(label_data):
        # Original start offset (in ms) of this sentence
        orig_start = sent_label["s"]
        new_label = {"orig_s": orig_start, "l": []}
        
        sent_text = []
        for tok in sent_label["l"]:
            new_tok = {
                "s": tok["s"] - orig_start,
                "e": tok["e"] - orig_start,
                "d": tok["d"].lower()
            }
            sent_text.append(tok["d"])
            new_label["l"].append(new_tok)
        
        # Dump new label for this sentence
        with open(os.path.join(NEW_LABEL_PATH, file_id + "-" + str(i) + ".json"), "wt") as f:
            f.write(json.dumps(new_label))
        
        # Dump new audio data
        sent_audiodata = get_audio_segment(audiodata, samplerate, sent_label["s"], sent_label["e"])

        sf.write(
            file=os.path.join(NEW_SONGS_PATH, file_id + "-" + str(i) + ".wav"),
            data=sent_audiodata,
            samplerate=samplerate
        )