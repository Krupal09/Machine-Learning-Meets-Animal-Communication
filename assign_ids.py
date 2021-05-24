import os
import uuid
from shutil import copyfile
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    dest="input_dir",
    type=str,
    help='Absolute path to the directory of the input data'
)
parser.add_argument(
    "--output_dir",
    dest="output_dir",
    type=str,
    help='Absolute path to the output directory'
)

args = parser.parse_args()

if __name__ == "__main__":

    data_path = args.input_dir
    output_path = args.output_dir

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(data_path):
        ids = []
        for file in files:
            new_id = str(uuid.uuid4().int)[:5]
            while new_id in ids:
                new_id = str(uuid.uuid4().int)[:5]
            ids.append(new_id)
        assert(len(files) == len(set(ids)))
        for i, file in enumerate(files):
            old_name = os.path.join(data_path, file)
            new_name = os.path.join(output_path, file.replace('_00000_', '_' + ids[i] + '_'))
            copyfile(old_name, new_name)
