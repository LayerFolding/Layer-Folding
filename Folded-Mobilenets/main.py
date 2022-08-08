
# Copyright 2021 Samsung. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import argparse
import os.path as osp

import tensorflow.compat.v1 as tf
from preprocessing import preprocess_image
import tf_slim as slim

import imagenet
import mobilenet_v2

parser = argparse.ArgumentParser(description='Evaluate folded mv2')
parser.add_argument('--dataset-dir', type=str, help='path to imagenet dir')
parser.add_argument('--dm', type=float,
                    help='The desired folded depth multiplier (0.75, 1., 1.4)')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='The directory where results will be saved')
parser.add_argument('--checkpoint-dir', type=str,
                    help='Path to the directory that contains the checkpoints')
DM2NETWORKFN = {
    1: mobilenet_v2.mobilenet_v2_1_folded,
    0.75: mobilenet_v2.mobilenet_v2_075_folded,
    1.4: mobilenet_v2.mobilenet_v2_140_folded
}

DM2CHECKPOINT = {
    1.4: 'mv2_140.ckpt',
    1: 'mv2_1.ckpt',
    0.75: 'mv2_075.ckpt'
}


def main(args):
    args = parser.parse_args(args)
    NUM_EVAL_IMAGES = 50000
    BATCH_SIZE = 1024
    eval_steps = NUM_EVAL_IMAGES // BATCH_SIZE
    with tf.Graph().as_default() as g:
        dataset = imagenet.get_split('validation', args.dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, shuffle=False
        )
        image, label = provider.get(['image', 'label'])
        image = preprocess_image(image, 224)
        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE
        )
        try:
            network_fn = DM2NETWORKFN[args.dm]
            ckpt_path = osp.join(args.checkpoint_dir, DM2CHECKPOINT[args.dm])
        except KeyError:
            raise KeyError('Unknown dm ({}) - possible values are 0.75, 1, 1.4'.format(args.dm))

        logits, _ = network_fn(images)
        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
        })

        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        tf_global_step = slim.get_or_create_global_step()
        variables_to_restore = slim.get_variables_to_restore()

        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=ckpt_path,
            logdir=args.logdir,
            num_evals=eval_steps,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore)


if __name__ == '__main__':
    main(sys.argv[1:])
