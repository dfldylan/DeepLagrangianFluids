#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_val
from fluid_evaluation_helper import FluidErrors
from datasets.fluidsnet_dataloader_loop import Datasets


def evaluate_whole_sequence_tf(model,
                               val_dataset,
                               frame_skip,
                               scale=1):
    print('evaluating.. ', end='')

    last_scene_id = None
    # time0 = time.time()
    for data in val_dataset:
        # scene_id = data['scene_id0'][0]
        # if last_scene_id is None or last_scene_id != scene_id:
        #     print(scene_id, end=' ', flush=True)
        #     last_scene_id = scene_id
        #     box = data['box'][0]
        #     box_normals = data['box_normals'][0]
        #     init_pos = data['pos0'][0]
        #     init_vel = data['vel0'][0]
        #
        #     inputs = (init_pos, init_vel, None, box, box_normals)
        # else:
        #     inputs = (pr_pos, pr_vel, None, box, box_normals)
        # print(scene_id, end=' ', flush=True)
        # last_scene_id = scene_id
        # time0 = time.time()
        box = data['box']
        box_normals = data['box_normals']
        init_pos = data['pos0']
        init_vel = data['vel0']
        inputs = (init_pos, init_vel, None, box, box_normals)
        val_dataset.write_csv(init_pos, init_vel)
        for step in range(val_dataset.pred_loop_num):
            pr_pos, pr_vel = model(inputs)
            val_dataset.fps += 1
            val_dataset.write_csv(pr_pos, pr_vel)
            inputs = (pr_pos, pr_vel, None, box, box_normals)

        # time1 = time.time()
        # val_dataset.save_time(time1-time0)
        # time0 = time.time()

        # frame_id = data['frame_id0'][0]
        # if frame_id > 0 :
        #     gt_pos = data['pos0'][0]

    print('done')

    return None


def eval_checkpoint(checkpoint_path, options, cfg):
    val_dataset = Datasets(seq_length=1)

    if checkpoint_path.endswith('.index'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model', {}))
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        checkpoint.restore(
            os.path.splitext(checkpoint_path)[0]).expect_partial()

        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip, **cfg.get('evaluation', {}))
    else:
        raise Exception('Unknown checkpoint format')


def print_errors(fluid_errors):
    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])
    print('====================\n', result)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a fluid network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainscript",
                        type=str,
                        required=True,
                        help="The python training script.")
    parser.add_argument("--cfg",
                        type=str,
                        required=True,
                        help="The path to the yaml config file")
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        required=False,
        help="The checkpoint iteration. The default is the last checkpoint.")
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="If set uses the specified weights file instead of a checkpoint.")
    parser.add_argument("--frame-skip",
                        type=int,
                        default=5,
                        help="The frame skip. Default is 5.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    global trainscript
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript = importlib.import_module(module_name)

    train_dir = module_name + '_' + os.path.splitext(os.path.basename(args.cfg))[0]
    # val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))

    # get a list of checkpoints

    # tensorflow checkpoints
    checkpoint_files = glob(
        os.path.join(train_dir, 'checkpoints', '*ckpt-*.index'))
    all_checkpoints = sorted([
        (int(re.match('.*ckpt-(\d+)\.(pt|index)', x).group(1)), x)
        for x in checkpoint_files
    ])

    # select the checkpoint
    checkpoint = all_checkpoints[-1]

    # output_path = train_dir + '_eval_{}.json'.format(checkpoint[0])
    print('evaluating :', checkpoint)
    eval_checkpoint(checkpoint[1], args, cfg)

    # print_errors(fluid_errors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
