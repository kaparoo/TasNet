# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from tasnet import TasNet, TasNetParam
from dataset import Dataset, DatasetParam

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', None,
                    "Directory to save weights", required=True)
flags.DEFINE_integer('epochs', None, 'Number of epochs.')
flags.DEFINE_integer('K', 20, 'Total number of segments in the input.')
flags.DEFINE_integer('L', 40, 'Length of each segment in the input.')
flags.DEFINE_integer('N', 500, 'Number of basis.')
flags.DEFINE_integer('H', 1000, 'Number of hidden units for each LSTM layer.')
flags.DEFINE_boolean('causal', False, 'Causality of the TasNet model.')
flags.DEFINE_string('dataset_path', f'{Path.home()}/musdb18', 'Dataset path')
flags.DEFINE_integer(
    'num_songs', 5, 'Number of songs to get sample from for each epoch.')
flags.DEFINE_integer('num_samples', 100, 'Number of samples.')
flags.DEFINE_integer('max_decoded', 100,
                     'The Maximum number of decoded songs in the memory.')
flags.DEFINE_integer('batch_size', 400, 'Number of batches for each epoch.')


def get_tasnet_param() -> TasNetParam:
    return TasNetParam(K=FLAGS.K, L=FLAGS.L, N=FLAGS.N, H=FLAGS.H,
                       causal=FLAGS.causal)


def get_dataset_param() -> DatasetParam:
    return DatasetParam(dataset_path=FLAGS.dataset_path,
                        num_songs=FLAGS.num_songs,
                        num_samples=FLAGS.num_samples,
                        num_segments=FLAGS.K,
                        len_segment=FLAGS.L,
                        batch_size=FLAGS.batch_size)


def main(argv):
    tasnet = TasNet.make(get_tasnet_param())
    dataset = Dataset(get_dataset_param())
    checkpoint_dir = FLAGS.checkpoint

    epoch = 0
    if path.exists(checkpoint_dir):
        checkpoints = [name for name in listdir(
            checkpoint_dir) if 'ckpt' in name]
        checkpoint_name = checkpoints[-1].split(".")[0]
        epoch = int(checkpoint_name) + 1
        tasnet.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    left_epochs = FLAGS.epochs
    while left_epochs == None or left_epochs > 0:
        print(f"Epoch: {epoch}")
        tasnet.fit(dataset.make_dataset())
        tasnet.save_weights(f"{checkpoint_dir}/{epoch:05d}.ckpt")
        epoch += 1
        if left_epochs != None:
            left_epochs -= 1
        tasnet.param.save(f"{checkpoint_dir}/config.txt")
        tasnet.save(f"{checkpoint_dir}/model")


if __name__ == '__main__':
    app.run(main)
