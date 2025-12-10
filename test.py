import torch
import torch.nn as nn
from model.basic_encoder import generate_encoder
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import logging
import json
import argparse
import os
import tqdm

from data.dataset import *
from model.harl_model import HARL
from loss_function import HARL_Loss
from model.PiPNet import generate_pip


class Tester():
    def __init__(self, checkpoint_path, EYEDIAP_root, MPII_root, target_domain="MPII", gnn_layers=4, batch_size=32):
        self.checkpoint_path = checkpoint_path
        self.EYEDIAP_root = EYEDIAP_root
        self.MPII_root = MPII_root
        self.target_domain = target_domain
        self.gnn_layers = gnn_layers
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_harl()

    def _init_harl(self):
        eye_encoder = generate_encoder(model_depth=18, n_classes=2, n_input_channels=1)
        # face_encoder = generate_encoder(model_depth=18, n_classes=2, n_input_channels=3)
        face_encoder = generate_pip()
        model = HARL(self.gnn_layers, eye_encoder, face_encoder)
        model.to(self.device)
        return model
    
    def gazeto3d(self, gaze):
        assert gaze.size == 2, "The size of gaze must be 2"
        gaze_gt = np.zeros([3])
        gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
        gaze_gt[1] = -np.sin(gaze[1])
        gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
        return gaze_gt

    def angular(self, pred, gt):
        total = np.sum(pred * gt)
        return np.arccos(min(total/(np.linalg.norm(pred)* np.linalg.norm(gt)), 0.9999999))*180/np.pi

    def init_testset(self):
        test_datasets = []
        if self.target_domain == "DIAP":
            for i in range(4):
                dataset = Eyediap_ds(self.EYEDIAP_root, i)
                test_datasets.append(dataset)
        elif self.target_domain == "MPII":
            for i in range(15):
                dataset = MPII_ds(self.MPII_root, i)
                test_datasets.append(dataset)
        else:
            raise ValueError

        return ConcatDataset(test_datasets)
    
    def load_cp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def test(self):
        self.load_cp(self.checkpoint_path)
        self.model.eval()
        count = 0
        acc_list = []
        test_set = self.init_testset()
        test_ld = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        for i, batchdata in enumerate(tqdm.tqdm(test_ld)):
            batchdata = {k: v.to(self.device) for k, v in batchdata.items()}
            outputs = self.model(batchdata)
            label = outputs["gt_face_gaze"]
            pred = outputs["face_gaze"]
            for k, gaze in enumerate(pred):
                gaze = gaze.cpu().detach().numpy()
                gt = label.cpu().numpy()[k]
                count += 1
                error = self.angular(self.gazeto3d(gaze),
                               self.gazeto3d(gt)
                               )
                acc_list.append(error)
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        print(f"Test Angular Error: Mean {mean_acc:.4f}, Std {std_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--target_domain", type=str, default="MPII", help="target domain for cross-domain training, choose from ['DIAP', 'MPII']")
    parser.add_argument("--MPII_root", type=str, default="/home/tqd/home/dataset/MPIIFaceGaze", help="path to MPII dataset")
    parser.add_argument("--EYEDIAP_root", type=str, default="/home/tqd/home/dataset/Eyediap", help="path to EYEDIAP dataset")
    parser.add_argument("--gnn_layers", type=int, default=4, help="number of GNN layers in HARL model's GFM")
    parser.add_argument("--checkpoint_path", "-cp", type=str, default='./', help="path to load checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    tester = Tester(checkpoint_path=args.checkpoint_path,
                    EYEDIAP_root=args.EYEDIAP_root,
                    MPII_root=args.MPII_root,
                    target_domain=args.target_domain,
                    gnn_layers=args.gnn_layers,
                    batch_size=args.batch_size)
    
    tester.test()