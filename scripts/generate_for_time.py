#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

test_dataset = r'/workdir/fluidmlp3/data/test_datasets_npz/sim_0001_0_50'
out_folder = r'../../data/fluidmlp3_pred'
output_dir = os.path.join(out_folder, os.path.split(test_dataset)[-1])
pred_frame_num = 50

def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if not vel is None:
        arrs['vel'] = vel
    np.savez(path_without_ext + '.npz', **arrs)


def run_sim_tf(trainscript_module, weights_path, num_steps, options):

    # init the network
    model = trainscript_module.create_model()
    model.init()
    model.load_weights(weights_path, by_name=True)

    box_npz = os.path.join(test_dataset, 'box.npz')
    fluid_npz = os.path.join(test_dataset, 'fluid_0000.npz')
    box_npz_data =  np.load(box_npz, allow_pickle=False)
    box, box_normals = box_npz_data['box'], box_npz_data['box_normals']
    fluid_npz_data = np.load(fluid_npz, allow_pickle=False)
    vel, pos = fluid_npz_data['vel'] ,fluid_npz_data['pos']

    # export static particles
    # write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)
    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

    s_time = time.time()
    for step in range(1000):
        if pos.shape[0]:
            # fluid_output_path = os.path.join(output_dir,
            #                                  'fluid_{0:04d}'.format(step))
            # if isinstance(pos, np.ndarray):
            #     write_particles(fluid_output_path, pos, vel, options)
            # else:
            #     write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
            #                     options)

            inputs = (pos, vel, None, box, box_normals)
            pos, vel = model(inputs)
    e_time = time.time()
    a_time = (e_time-s_time)/(1000)
    print(a_time)
        # # remove out of bounds particles
        # if step % 10 == 0:
        #     print(step, 'num particles', pos.shape[0])
        #     mask = pos[:, 1] > min_y
        #     if np.count_nonzero(mask) < pos.shape[0]:
        #         pos = pos[mask]
        #         vel = vel[mask]

def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainscript",
                        type=str,
                        default='train_network_tf.py',
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        default='pretrained_model_weights.h5',
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=50,
                        help="The number of simulation steps. Default is 250.")

    args = parser.parse_args()
    print(args)

    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript_module = importlib.import_module(module_name)

    # os.makedirs(output_dir)

    return run_sim_tf(trainscript_module, args.weights,
                          args.num_steps, args)

if __name__ == '__main__':
    sys.exit(main())
