import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def return_transform(mono=True, Face=False, HQ=True):
    assert mono != Face
    if Face:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif mono:
        if HQ:
            return transforms.Compose([
                transforms.CenterCrop((400, 400)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop((36, 36)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

def GazeTo2d(gaze):
    yaw = np.arctan2(gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])

    return np.array([yaw, pitch])


class OpenEDS_ds(Dataset):
    def __init__(self, OpenEDS_root, split='train'):
        super().__init__()
        self.root = OpenEDS_root
        self.split = split
        
        assert self.split in ['train', 'validation']
        self.eye_image_root = os.path.join(os.path.join(self.root, self.split), 'sequences')
        self.eye_meta = os.path.join(self.root, 'metadata')
        self.eye_data = pd.read_csv(os.path.join(self.eye_meta, split) + '.csv')
        self.transform = return_transform(mono=True, Face=False, HQ=True)
    def __len__(self):
        return len(self.eye_data)
    
    def __getitem__(self, i):
        sample = {}
        eye_image_name = self.eye_data.iloc[i, self.eye_data.columns.get_loc('image')]
        eye_image_root = os.path.join(self.eye_image_root, eye_image_name) + '.png'
        eye_image_root = eye_image_root.replace('\\', '/')
        eye_image = Image.open(eye_image_root)
        eye_image = self.transform(eye_image)

        eye_gaze3d = self.eye_data.iloc[i, self.eye_data.columns.get_loc('gaze_gt_vec')]
        eye_gaze3d = eye_gaze3d.strip('[]')
        eye_gaze3d = [x.strip() for x in eye_gaze3d.split() if x]
        eye_gaze3d = np.array([float(x) for x in eye_gaze3d])
        eye_gaze2d = GazeTo2d(eye_gaze3d)
        eye_gaze2d = torch.tensor(eye_gaze2d, dtype=torch.float32)
        eye_gaze3d = torch.tensor(eye_gaze3d, dtype=torch.float32)

        sample['hq_mono'] = eye_image
        sample['hq_gaze2d'] = eye_gaze2d
        sample['hq_gaze3d'] = eye_gaze3d

        return sample

class Eyediap_ds(Dataset):
    def __init__(self, EYEDIAP_root, cluster):
        super().__init__()

        self.face_root = EYEDIAP_root
        self.transform_face = return_transform(mono=False, Face=True)
        self.transform_mono = return_transform(mono=True, Face=False, HQ=False)
 
        self.face_image_root = os.path.join(self.face_root, 'Image')
        self.face_label_root = os.path.join(self.face_root, 'Label')

        label_path = os.path.join(self.face_root, f'Cluster{cluster}.label')
        self.face_data = pd.read_csv(label_path, delimiter=' ')

    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, i):
        sample={}
        face_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Face')]
        face_image = Image.open(os.path.join(self.face_image_root, face_image_name))
        left_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Left')]
        left_image = Image.open(os.path.join(self.face_image_root, left_image_name))
        right_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Right')]
        right_image = Image.open(os.path.join(self.face_image_root, right_image_name))

        face_gaze2d = self.face_data.iloc[i, self.face_data.columns.get_loc('2DGaze')]
        face_gaze2d = np.array([float(x) for x in face_gaze2d.split(',')])
        face_gaze2d = torch.tensor(face_gaze2d, dtype=torch.float32)

        face_image_t = self.transform_face(face_image)
        left_image_t = self.transform_mono(left_image)
        right_image_t = self.transform_mono(right_image)

        sample['face'] = face_image_t
        sample['left'] = left_image_t
        sample['right'] = right_image_t
        sample['face_gaze2d'] = face_gaze2d
        return sample

class MPII_ds(Dataset):
    def __init__(self, MPII_root, cluster):
        super().__init__()
        
        self.face_root = MPII_root
        self.transform_face = return_transform(mono=False, Face=True)
        self.transform_mono = return_transform(mono=True, Face=False, HQ=False)
    
        self.face_image_root = os.path.join(self.face_root, 'Image')
        self.face_label_root = os.path.join(self.face_root, 'Label')


        label_path = os.path.join(self.face_label_root, f'p{cluster:02d}.label')
        self.face_data = pd.read_csv(label_path, delimiter=' ')


    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, i):
        sample={}
        face_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Face')]
        face_image = Image.open(os.path.join(self.face_image_root, face_image_name))
        left_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Left')]
        left_image = Image.open(os.path.join(self.face_image_root, left_image_name))
        right_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Right')]
        right_image = Image.open(os.path.join(self.face_image_root, right_image_name))

        face_gaze2d = self.face_data.iloc[i, self.face_data.columns.get_loc('2DGaze')]
        face_gaze2d = np.array([float(x) for x in face_gaze2d.split(',')])
        face_gaze2d = torch.tensor(face_gaze2d, dtype=torch.float32)

        face_image_t = self.transform_face(face_image)
        left_image_t = self.transform_mono(left_image)
        right_image_t = self.transform_mono(right_image)

        sample['face'] = face_image_t
        sample['left'] = left_image_t
        sample['right'] = right_image_t
        sample['face_gaze2d'] = face_gaze2d
        return sample

class Gaze360_ds(Dataset):
    def __init__(self, Gaze360_root, split='train'):
        super().__init__()

        self.face_root = Gaze360_root
        self.transform_face = return_transform(mono=False, Face=True, HQ=False)
        self.transform_mono = return_transform(mono=True, Face=False, HQ=False)
        self.split = split

        self.face_image_root = os.path.join(self.face_root, 'Image')
        self.face_label_root = os.path.join(self.face_root, 'Label')

        file_path = os.path.join(self.face_label_root, f'{split}.label')
        self.face_data = pd.read_csv(file_path, delimiter=' ')

    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, i):
        sample={}
        face_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Face')]
        face_image = Image.open(os.path.join(self.face_image_root, face_image_name))
        left_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Left')]
        left_image = Image.open(os.path.join(self.face_image_root, left_image_name))
        right_image_name = self.face_data.iloc[i, self.face_data.columns.get_loc('Right')]
        right_image = Image.open(os.path.join(self.face_image_root, right_image_name))

        face_gaze2d = self.face_data.iloc[i, self.face_data.columns.get_loc('2DGaze')]
        face_gaze2d = np.array([float(x) for x in face_gaze2d.split(',')])
        face_gaze2d = torch.tensor(face_gaze2d, dtype=torch.float32)

        face_image_t = self.transform_face(face_image)
        left_image_t = self.transform_mono(left_image)
        right_image_t = self.transform_mono(right_image)

        sample['face'] = face_image_t
        sample['left'] = left_image_t
        sample['right'] = right_image_t
        sample['face_gaze2d'] = face_gaze2d
        return sample



if __name__ == '__main__':
    pass
