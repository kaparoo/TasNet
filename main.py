from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir
from tasnet import TasNet, TasNetParam

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', None,
                    "Directory to save weights", required=True)
flags.DEFINE_string('dataset_path', f'{Path.home()}/musdb18', 'Dataset path')
flags.DEFINE_integer('epochs', None, 'Number of epochs.')
flags.DEFINE_integer('K', 20, 'Total number of segments in the input.')
flags.DEFINE_integer('L', 40, 'Length of each segment in the input.')
flags.DEFINE_integer('N', 500, 'Number of basis.')
flags.DEFINE_integer('H', 1000, 'Number of hidden units for each LSTM layer.')
flags.DEFINE_boolean('causal', False, 'Causality of the TasNet model.')


def get_model_param() -> TasNetParam:
    return TasNetParam(K=FLAGS.K, L=FLAGS.L, N=FLAGS.N, H=FLAGS.H,
                       causal=FLAGS.causal)


def main(argv):
    model_param = get_model_param()
    print(model_param.get_config())


if __name__ == '__main__':
    app.run(main)
