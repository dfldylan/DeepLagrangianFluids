import os
import pandas as pd
import numpy as np
from hash import hash
import shutil

# root_path = r'/root/datasets/mountain'
# log_path = os.path.join(root_path, 'log.csv')
# pred_fps = [350, 450]
output_folder_root = r'/root/datasets/pred/cconv'


class Datasets(object):
    def __init__(self, seq_length=10):
        # self.df = pd.read_csv(log_path)
        self.seq_length = seq_length
        self.index = None
        self.start_csv_list = [r'/root/datasets/large/all_particles_350.csv',
                               r'/root/datasets/large/all_particles_390.csv',
                               r'/root/datasets/normal/0/all_particles_300.csv',
                               r'/root/datasets/normal/171/all_particles_300.csv',
                               r'/root/datasets/normal/175/all_particles_300.csv']
        self.pred_loop_num = None

    def __len__(self):
        return 1

    def __iter__(self):
        for i in range(len(self.start_csv_list)):
            self.start_csv = self.start_csv_list[i]
            self.box = pd.read_csv(self.start_csv)
            self.box = self.box[self.box.isFluidSolid == 1].iloc[:, :3].values
            self.box_normals = self.cal_box_normals(self.box)
            if i<2:
                self.pred_loop_num=30
            else:
                self.pred_loop_num=50
            start_csv = self.start_csv

            # prepare
            self.fps = int(start_csv.split(r'/')[-1].split(r'.')[0].split(r'_')[-1])
            id = hash(start_csv)
            print(id, start_csv)
            self.pred_folder = r'./pred/' + id
            self.output_folder = os.path.join(output_folder_root, id)
            os.makedirs(self.pred_folder, exist_ok=True)
            os.makedirs(self.output_folder, exist_ok=True)

            log_data = pd.read_csv(start_csv, dtype=float)
            data = log_data[log_data.isFluidSolid == 0].iloc[:, :6].values
            # data_s = log_data[log_data.isFluidSolid == 1].iloc[:, :6].values

            sample = {
                'pos0': data[:, :3],
                'vel0': data[:, 3:],
                'box': self.box,
                'box_normals': self.box_normals
            }
            yield sample

    def cal_box_normals(self, box):
        box = box / np.max(box, axis=0,keepdims=True)
        box = box.astype(np.int).astype(np.float64)
        box = 0 - box
        box_norm = np.linalg.norm(box, axis=1, keepdims=True)
        box /= box_norm
        return box

    def write_csv(self, pos, vel):
        # concat csv data with solid particles
        df = pd.read_csv(self.start_csv)
        df = df[df.isFluidSolid == 1]
        data_fluid = np.concatenate((pos, vel), axis=1)  # [-1, 6]
        df0 = pd.DataFrame(data_fluid, columns=df.columns[:6])
        df0[df.columns[6:7]] = 0.004
        df0[df.columns[7:18]] = 0
        df = df.append(df0)

        print(str(self.fps) + ' ok!')
        # write csv -- fast mode start
        df.to_csv(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), index=False)
        if self.output_folder is not None:
            shutil.copy(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), self.output_folder)
