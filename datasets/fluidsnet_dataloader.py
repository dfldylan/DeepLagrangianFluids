import os
import pandas as pd
import numpy as np
import dataflow

root_path = r'/root/datasets/normal'


class Dataset(dataflow.RNGDataFlow):
    def __init__(self, root=root_path, seq_length=3, random_rotation=True):
        self.seq_length = seq_length
        self.root = root
        self.folder_path, self.down, self.up = find_files(root, seq_length, range_down=300, range_up=500)
        self.box = pd.read_csv(os.path.join(root_path, '0/all_particles_1.csv'))
        self.box = self.box[self.box.isFluidSolid == 1].iloc[:, :3].values
        self.box_normals = self.cal_box_normals(self.box)
        self.random_rotation = random_rotation

    def __len__(self):
        return len(self.folder_path) * (self.up - self.down)

    def __getitem__(self, _):
        folder = np.random.choice(self.folder_path)
        start_fps = np.random.randint(self.down, self.up)
        print(folder, str(start_fps), end=' ')
        log_data_list = []
        for i in range(self.seq_length):
            fps = start_fps + i
            path = os.path.join(folder, "all_particles_" + str(fps) + ".csv")
            log_data = pd.read_csv(path, dtype=float)
            log_data = log_data[log_data.isFluidSolid==0].iloc[:, :6].values
            log_data_list.append(log_data)
            print(str(log_data.shape[0]), end=' ')
        print('')
        try:
            stack = np.stack(log_data_list, axis=0)
            # if stack.shape[1] > 46000:
            #     raise
            return stack
        except:
            print('again')
            return self.__getitem__(0)

    def __iter__(self):
        while True:
            data = self.__getitem__(0)
            sample = {
                'pos0': data[0, :, :3],
                'vel0': data[0, :, 3:6],
                'pos1': data[1, :, :3],
                'pos2': data[2, :, :3],
                'box': self.box,
                'box_normals': self.box_normals
            }
            if self.random_rotation:
                angle_rad = self.rng.uniform(0, 2 * np.pi)
                s = np.sin(angle_rad)
                c = np.cos(angle_rad)
                rand_R = np.array([c, 0, s, 0, 1, 0, -s, 0, c],
                                  dtype=np.float32).reshape((3, 3))
                for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
                    sample[k] = np.matmul(sample[k], rand_R).astype(np.float32)
            yield sample

    def cal_box_normals(self, box):
        box = box / np.array([[4, 2, 4]])
        box = box.astype(np.int).astype(np.float64)
        box = 0 - box
        box_norm = np.linalg.norm(box, axis=1, keepdims=True)
        box /= box_norm
        return box


def find_files(root_path, seq_length, range_up=None, range_down=None, scene_num=None):
    folders = []
    exist_folders = [item for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    if scene_num is not None:
        for each in scene_num:
            each = str(each)
            if each in exist_folders:
                folders.append(each)
            else:
                print("folder " + each + " doesn't exist!")
    else:
        folders = exist_folders

    folder_path = [os.path.join(root_path, item) for item in folders]
    down = 1 if range_down is None else range_down
    up = 1500 if range_up is None else range_up
    up -= seq_length
    if up <= down:
        print('failed')
        exit(-1)
    return folder_path, down, up


def get_fps(file):
    return int(file.split(r'.')[0].split(r'_')[-1])
