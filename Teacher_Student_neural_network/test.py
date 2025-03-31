"""
File: test.py
Original Author: Yuqin Yuan
Date: 2024-08-28
"""

import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import STUDNN
from dataset import RadarDataset
from sklearn.metrics import f1_score, accuracy_score 
import pkg_resources
# 获取已安装包的信息
installed_packages = pkg_resources.working_set

# 打印已安装包的名称和版本
for package in installed_packages:
    print(f"{package.key}=={package.version}")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data', help='Directory to data dir')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='models/net.pkl', help='Path to saved model')
    parser.add_argument('--thresholds-path', type=str, default='models/thresholds.mat', help='Path to saved model')
    return parser.parse_args()

def cal_scores(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # cx = [tn, fp, fn, tp]
    print('TN:',tn, 'FP:', fp, 'FN:', fn, 'TP :',tp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    print('F1s:',f1)
    print('Accs:', acc)
    print('sen:', sen)
    print('spe:', spe)

def evaluate(dataloader, device, net,args, threshold):
    net.eval()
    thresholds = list()
    thresholds.append(threshold*3)
    output_list, labels_list, patient_list= [], [], []
    with torch.no_grad():
        for data in enumerate(tqdm(dataloader)):
            usedata = data[1]['Radar'].cuda()
            labels = data[1]['label'].cuda()
            subject_id = data[1]['subject_id']
            output = net(usedata)
            output = torch.sigmoid(output)
            output_list.append(output.data.cpu().numpy())
            labels_list.append(labels.data.cpu().numpy())
            patient_list.append(np.array(subject_id))
        # sequence-level strategy
        y_trues = np.vstack(labels_list)
        y_scores = (np.vstack(output_list))*3
        patient_names = np.vstack(patient_list)
        y_preds = (y_scores >= thresholds[0]).astype(int)
        print('Test set prediction results using sequence-level strategy')
        cal_scores(y_trues,y_preds)
        print('thresholds',thresholds)
        # set-level strategy
        str_list = patient_names.tolist()
        result_dict = {}
        for idx, string in enumerate(str_list):
            key = string[0][:-2]
            if key in result_dict:
                result_dict[key].append(y_preds[idx])
            else:
                result_dict[key] = [y_trues[idx]]
                result_dict[key].append(y_preds[idx])
        person_trues = list()
        person_predicteds = list()
        for key, result in result_dict.items():
            person_trues.append(result[0])
            if np.sum(np.array(result[1:])) >= 2:
                person_predicted = 1
                person_predicteds.append(person_predicted)
            else:
                person_predicted = 0
                person_predicteds.append(person_predicted)
        print('Test set prediction results using set-level strategy')
        cal_scores(person_trues,person_predicteds)
                
   
if __name__ == "__main__":
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    data_dir = args.data_dir
    label_csv = os.path.join(data_dir, 'test_labels.csv')
    net = STUDNN.STUDNN(num_classes=1).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    threshold = sio.loadmat(args.thresholds_path)
    threshold = np.squeeze(threshold['thresholds'])
    threshold = threshold.tolist()
    test_dataset = RadarDataset('test', data_dir, label_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('Results on test data:')
    evaluate(test_loader, device, net,args,threshold)
