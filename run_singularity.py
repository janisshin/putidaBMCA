# adapted from https://github.com/EvanKomp/alphafold/blob/main/run_singularity.py
"""
Singularity launch script for putida bmca Singularity image.
To be run on hyak command line
"""

import os
import pathlib
from typing import Tuple

from absl import app
from absl import flags
from absl import logging
from spython.main import Client

import tempfile
import subprocess

#### USER CONFIGURATION ####

# Path to Singularity image. This relies on
# the environment variable ALPHAFOLD_DIR which is the
# directory where AlphaFold is installed.
singularity_image = Client.load('/gscratch/cheme/jshin1/putidabmca.sif')

# tmp directory
if 'TMP' in os.environ:
    tmp_dir = os.environ['TMP']
elif 'TMPDIR' in os.environ:
    tmp_dir = os.environ['TMPDIR']
else:
    tmp_dir = './tmp'

# Default path to a directory that will store the results.
output_dir_default = tempfile.mkdtemp(dir=tmp_dir, prefix=None)

logging.info(f'INFO: tmp_dir = {tmp_dir}')
logging.info(f'INFO: output_dir_default = {output_dir_default}')

#### END USER CONFIGURATION ####
flags.DEFINE_string(
    'output_dir', output_dir_default,
    'Path to a directory that will store the results.')
flags.DEFINE_string(
    'run_name', None,
    'Name of the resulting pickle file.')
flags.DEFINE_integer(
    'iter', None,
    'Number of iterations for the PyMC ADVI fitting ')

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  binds = []
  command_args = []
  
  output_target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, 'output')
  binds.append(f'{FLAGS.output_dir}:{output_target_path}')
  logging.info('Binding %s -> %s', FLAGS.output_dir, output_target_path)

  tmp_target_path = '/tmp'
  binds.append(f'{tmp_dir}:{tmp_target_path}')
  logging.info('Binding %s -> %s', tmp_dir, tmp_target_path)

  command_args.extend([
      f'--output_dir={output_target_path}',
      f'--runName={FLAGS.max_template_date}', ## flag or user input?
      f'--iter={FLAGS.max_template_date}', ## flag or user input?
      '--logtostderr',
  ])

  options = [
    '--bind', f'{",".join(binds)}',
  ]

  # Run the container.
  # Result is a dict with keys "message" (value = all output as a single string),
  # and "return_code" (value = integer return code)
  result = Client.run(
            singularity_image,
            command_args,
            return_result=True,
            options=options
        )

if __name__ == '__main__':
  flags.mark_flags_as_required(['run_name'])
  app.run(main)
