# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import youtube_dl

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from tasnet import TasNet, TasNetParam
from dataset import Dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None,
                    "Directory containing saved weights", required=True)
flags.DEFINE_string("video_id", None, "YouTube video ID", required=True)


def youtube_dl_hook(d):
    if d["status"] == "finished":
        print("Done downloading...")


def main(argv):
    checkpoint_dir = FLAGS.checkpoint
    if not path.exists(checkpoint_dir):
        raise ValueError(f"'{checkpoint_dir}' does not exist")

    checkpoints = [name for name in listdir(checkpoint_dir) if "ckpt" in name]
    if not checkpoints:
        raise ValueError(f"No checkpoint exists")
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]

    param = TasNetParam.load(f"{checkpoint_dir}/config.txt")
    print(param.get_config)
    tasnet = TasNet.make(param)
    tasnet.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    video_id = FLAGS.video_id

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "44100",
        }],
        "outtmpl": "%(title)s.wav",
        "progress_hooks": [youtube_dl_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_id, download=False)
        status = ydl.download([video_id])

    title = info.get("title", None)
    filename = title + ".wav"
    audio, sr = librosa.load(filename, sr=44100, mono=True)

    num_samples = audio.shape[0]
    num_portions = num_samples // (param.K * param.L)
    num_samples = num_portions * (param.K * param.L)

    print("predicting...")

    audio = audio[:num_samples]
    audio = np.reshape(audio, [num_portions, param.K, param.L])

    separated = tasnet.predict(audio)
    separated = np.transpose(separated, (1, 0, 2, 3))
    separated = np.reshape(separated, (param.C, num_samples))

    print("saving...")

    for idx, stem in enumerate(Dataset.STEMS):
        sf.write(f"{title}_{stem}.wav", separated[idx], sr)


if __name__ == '__main__':
    app.run(main)
