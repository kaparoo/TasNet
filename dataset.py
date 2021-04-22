# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import musdb
import random
import gc
from tqdm import tqdm


class DatasetParam:
    __slots__ = 'dataset_path', 'num_songs', 'num_samples', 'num_segments', 'len_segment', 'batch_size'

    def __init__(self,
                 dataset_path: str = None,
                 num_songs: int = 100,
                 num_samples: int = 100,
                 num_segments: int = 20,
                 len_segment: int = 40,
                 batch_size: int = 400,):
        self.dataset_path = dataset_path
        self.num_songs = num_songs
        self.num_samples = num_samples
        self.num_segments = num_segments
        self.len_segment = len_segment
        self.batch_size = batch_size


class DecodedTrack:
    __slots__ = 'length', 'mixed', 'stems'

    @staticmethod
    def from_track(track):
        mixed = (track.audio[:, 0], track.audio[:, 1])
        length = mixed[0].shape[-1]
        stems = {}
        for stem in Dataset.STEMS:
            audio = track.targets[stem].audio
            stems[stem] = (audio[:, 0], audio[:, 1])
        return DecodedTrack(length, mixed, stems)

    def __init__(self, length, mixed, stems):
        self.length = length
        self.mixed = mixed
        self.stems = stems


class Dataset:

    STEMS = 'vocals', 'drums', 'bass', 'other'

    def __init__(self, param: DatasetParam, musdb_subsets: str = 'train', max_decoded: int = 100, **kwargs):
        if max_decoded < 1:
            raise ValueError("max_decoded must be greater than 0!")
        self.param = param
        self.tracks = list(
            musdb.DB(root=param.dataset_path, subsets=musdb_subsets))
        self.num_tracks = len(self.tracks)
        self.decoded = [None] * self.num_tracks
        self.num_decoded = 0
        self.max_decoded = max_decoded
        self.ord_decoded = [-1] * self.num_tracks
        self.next_ord = 0

    def decode(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        if len(indices) > self.max_decoded:
            raise ValueError("Cannot decode more then `max_decoded` tracks!")

        if indices != None:
            print(f"Decoding Audio {indices}...")
            for index in tqdm(indices):
                if self.num_decoded == self.max_decoded:
                    index = np.argmin(self.ord_decoded)
                    self.decoded[index] = None
                    self.num_decoded -= 1
                    self.ord_decoded[index] = -1
                    gc.collect()
                self.decoded[index] = DecodedTrack.from_track(
                    self.tracks[index])
                self.num_decoded += 1
                self.ord_decoded[index] = self.next_ord
                self.next_ord += 1

    def generate(self):
        indices = list(range(self.num_tracks))
        random.shuffle(indices)
        indices = indices[:self.param.num_songs]
        self.decode(indices)

        duration = self.param.num_segments * self.param.len_segment

        for _ in range(self.param.batch_size):
            x_batch = np.zeros([self.param.num_samples * 2,
                                self.param.num_segments,
                                self.param.len_segment])
            y_batch = np.zeros([self.param.num_samples * 2,
                                len(Dataset.STEMS),
                                self.param.num_segments,
                                self.param.len_segment])

            for i in range(self.param.num_samples):
                track = self.decoded[random.choice(indices)]
                start = random.randint(0, track.length - duration)

                for j in range(self.param.num_segments):
                    left = i * 2
                    right = left + 1
                    begin = start + j * self.param.len_segment
                    end = begin + self.param.len_segment
                    x_batch[left][j] = track.mixed[0][begin:end]
                    x_batch[right][j] = track.mixed[1][begin:end]

                    for c, stem in enumerate(Dataset.STEMS):
                        y_batch[left][c][j] = track.stems[stem][0][begin:end]
                        y_batch[right][c][j] = track.stems[stem][1][begin:end]

            yield x_batch, y_batch

    def make_dataset(self) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape([self.param.num_samples * 2,
                                         self.param.num_segments,
                                         self.param.len_segment]),
                         tf.TensorShape([self.param.num_samples * 2,
                                         len(Dataset.STEMS),
                                         self.param.num_segments,
                                         self.param.len_segment]))
        return tf.data.Dataset.from_generator(lambda: self.generate(),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
