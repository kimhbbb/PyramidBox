import os
import scipy.io
import json
from torch.utils.data import Dataset
from PIL import Image
import torch

class WIDERFaceDataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.data_folder = data_folder

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxe'])

        return image, boxes

    def collate_fn(self):


        pass

def create_data_lists(wider_root_path, wider_label_path, output_path):
    wider_root_path = os.path.abspath(wider_root_path)
    wider_label_path = os.path.abspath(wider_label_path)

    make_file(wider_root_path, wider_label_path, output_path, mode='train')
    make_file(wider_root_path, wider_label_path, output_path, mode='test')

def make_file(wider_root_path, wider_label_path, output_path, mode = 'train'):
    if mode == 'train':
        f = scipy.io.loadmat(os.path.join(wider_label_path, 'wider_face_train.mat'))
        path_to_image = os.path.join(wider_root_path, 'WIDER_train/images')
    elif mode == 'test':
        f = scipy.io.loadmat(os.path.join(wider_label_path, 'wider_face_test.mat'))
        path_to_image = os.path.join(wider_root_path, 'WIDER_test/images')

    event_list = f.get('event_list')
    file_list = f.get('file_list')
    face_bbx_list = f.get('face_bbx_list')

    img_ids = list()
    event_ids = list()
    train_objects = list()

    for event_idx, event in enumerate(event_list):
        directory = event[0][0]
        for img_idx, img in enumerate(file_list[event_idx][0]):
            img_name = img[0][0]

            if mode == 'test':
                img_ids.append(os.path.join(path_to_image, directory, img_name + '.jpg'))
                # event_ids.append(directory)
                continue
            
            face_bbx = face_bbx_list[event_idx][0][img_idx][0]
            bboxes = []
            for i in range(face_bbx.shape[0]):
                if face_bbx[i][2] < 2 or face_bbx[i][3] < 2 or face_bbx[i][0] < 0 or face_bbx[i][1] < 0:
                        continue
                xmin = int(face_bbx[i][0]) -1
                ymin = int(face_bbx[i][1]) -1
                xmax = int(face_bbx[i][2]) -1
                ymax = int(face_bbx[i][3]) -1
                bboxes.append([xmin, ymin, xmax, ymax])

            if (len(bboxes) == 0):
                continue

            img_ids.append(os.path.join(path_to_image, directory, img_name + '.jpg') )
            # event_ids.append(directory)
            train_objects.append({'boxes': bboxes})

    if mode == 'train':
        with open(os.path.join(output_path, 'Train_images.json'), 'w') as j:
            json.dump(img_ids, j)
        with open(os.path.join(output_path, 'Train_objects.json'), 'w') as j:
            json.dump(train_objects, j)

if __name__ == '__main__':
    create_data_lists(wider_root_path = './data',
                      wider_label_path = './data/wider_face_split',
                      output_path = './PyramidBox')