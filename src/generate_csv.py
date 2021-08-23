'''
    This script creates a clean CSV with image names and labels based on an original CSV.
    Another goal is to clean up and standardise the format for further processing.
    It also optionally splits the data into train and test sets based on the given split ratio.

    Author: David Kostka
    Date: 15.02.2021

    python src/generate_csv.py --csv_dir=data/datasets/raw/simulator --output_dir=data/datasets/simulator/labels
'''
import os
import csv
import pandas as pd
from absl import app, flags
import util.utils as util

FLAGS = flags.FLAGS
flags.DEFINE_string('csv_dir', '', 'Path to labels CSV directory')
flags.DEFINE_string('output_dir', '', 'Path to clean output CSV directory')

out_columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']


def remove_trailing_commas(csv_path):
    lines = []

    with open(csv_path, encoding='utf-8') as f_input:
        for line in f_input:
            line = line[:-1]
            line = line.split(',')[:8]
            lines.append(line)

    with open(csv_path, 'w', encoding='utf-8') as f_output:
        csv.writer(f_output, delimiter=',').writerows(lines)


def create_clean_csv(csv_path, output_path, in_columns, class_table):
    # Might be needed if the CSV has more commas than columns
    # TODO: Funktion auslagern in eigenen Script
    # remove_trailing_commas(csv_path)

    raw_labels = pd.read_csv(csv_path, sep=r'\s*,\s*', index_col=None, engine='python', usecols=in_columns).fillna(0)
    raw_labels = raw_labels.rename(columns=dict(zip(in_columns, out_columns)))

    raw_labels['class'] = raw_labels['class'].map(class_table)
    # Optional: Filter out ball examples
    # labels = raw_labels[(raw_labels['class'] == 1) | (raw_labels['class'] == 0)]
    if raw_labels[(raw_labels['class'] == 2)].empty: 
        print('-----------' + csv_path + '-------------')
        return
    labels = raw_labels
    return labels.to_csv(output_path, index=False)


def main(_):
    util.create_dirs([FLAGS.output_dir])

    # For simulator data
    #  in_columns = ['name', 'minX', 'minY', 'maxX', 'maxY', 'type']
    # class_table = {4: 1}

    # For ImageTagger data
    class_table = {0: 0, 'robot': 1, 'ball': 2}
    in_columns = out_columns

    for root, _, files in os.walk(FLAGS.csv_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                out_path = os.path.basename(root) + '.csv'
                out_path = os.path.join(FLAGS.output_dir, out_path)
                create_clean_csv(csv_path, out_path, in_columns, class_table)
                print('Created clean CSV in: ' + out_path)
                
    print('Done creating clean CSVs in: ' + FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)