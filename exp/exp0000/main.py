# basic
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import zipfile
import random
from tqdm import tqdm
from collections import OrderedDict, Counter
import lmdb
import six
from PIL import Image
from typing import List, Dict, Union, Tuple, Any
import cv2

# hydra
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize_config_dir

# wandb
import wandb

# sklearn
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold

# pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from utils import seed_everything, AverageMeter, get_lr, validation_metrics, CTCLabelConverter


def split_data(cfg, train_csv_path):
    '''
        StratifiedGroupKFold
            group: participant_id
            target: phrase_type (url, telephone, address)
    '''

    train_df = pd.read_csv(train_csv_path)

    num_set = set([str(i) for i in range(10)])
    tel_set = set(['-', '+'])
    url_set = set(['/', '.'])

    def check_phrase_type(phrase):
        char_set = set(phrase)
        if not (char_set - (num_set | tel_set)):
            return 'telephone'
        elif (char_set & url_set):
            return 'url'
        else:
            return 'address'

    train_df['phrase_type'] = train_df['phrase'].map(check_phrase_type)

    train_df['fold'] = -1
    for fold, (_, valid_fold_indices) \
            in enumerate(StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed).split(
            train_df.index, train_df['phrase_type'], groups=train_df['participant_id'])):

        train_df.loc[valid_fold_indices, 'fold'] = fold

    return train_df


def get_indices(cfg):
    landmark_dict = dict(
        silhouette=[
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],

        lips_upper_outer=[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        lips_lower_outer=[146, 91, 181, 84, 17, 314, 405, 321, 375],
        lips_upper_inner=[78, 191, 80, 81,
                          82, 13, 312, 311, 310, 415, 308],
        lips_lower_inner=[95, 88, 178, 87, 14, 317, 402, 318, 324],

        eye_right_upper0=[246, 161, 160, 159, 158, 157, 173],
        eye_right_lower0=[33, 7, 163, 144, 145, 153, 154, 155, 133],
        eye_right_upper1=[247, 30, 29, 27, 28, 56, 190],
        eye_right_lower1=[130, 25, 110, 24, 23, 22, 26, 112, 243],
        eye_right_upper2=[113, 225, 224, 223, 222, 221, 189],
        eye_right_lower2=[226, 31, 228, 229, 230, 231, 232, 233, 244],
        eye_right_lower3=[143, 111, 117, 118, 119, 120, 121, 128, 245],

        eye_brow_right_upper=[156, 70, 63, 105, 66, 107, 55, 193],
        eye_brow_right_lower=[35, 124, 46, 53, 52, 65],

        eye_iris_right=[473, 474, 475, 476, 477],

        eye_left_upper0=[466, 388, 387, 386, 385, 384, 398],
        eye_left_lower0=[263, 249, 390, 373, 374, 380, 381, 382, 362],
        eye_left_upper1=[467, 260, 259, 257, 258, 286, 414],
        eye_left_lower1=[359, 255, 339, 254, 253, 252, 256, 341, 463],
        eye_left_upper2=[342, 445, 444, 443, 442, 441, 413],
        eye_left_lower2=[446, 261, 448, 449, 450, 451, 452, 453, 464],
        eye_left_lower3=[372, 340, 346, 347, 348, 349, 350, 357, 465],

        eye_brow_left_upper=[383, 300, 293, 334, 296, 336, 285, 417],
        eye_brow_left_lower=[265, 353, 276, 283, 282, 295],

        eye_iris_left=[468, 469, 470, 471, 472],

        midway_between_eye=[168],

        nose_tip=[1],
        nose_bottom=[2],
        nose_right=[98],
        nose_left=[327],

        cheek_right=[205],
        cheek_left=[425],

        left_hand=list(range(468, 489)),
        right_hand=list(range(522, 543)),

        body=[11, 23, 24, 12],
        left_arm=[501, 503, 505],
        right_arm=[500, 502, 504],
    )

    use_types = cfg.use_types

    use_landmarks = {}
    for use_type in use_types:
        use_landmarks[use_type] = []
        for key, value in landmark_dict.items():
            if use_type in key:
                use_landmarks[use_type].extend(value)
    return use_landmarks

# dataset


class Asl2Dataset(Dataset):
    def __init__(self, cfg, df, lmdb_dir, char_to_idx, use_landmarks):
        self.cfg = cfg
        self.df = df
        self.env = lmdb.open(str(lmdb_dir), max_readers=32,
                             readonly=True, lock=False, readahead=False, meminit=False)
        self.array_dict = use_landmarks
        self.char_to_idx = char_to_idx

        self.hand_max_length = cfg.hand_max_length
        self.lips_max_length = cfg.lips_max_length
        self.padding = cfg.padding
        self.padding_value = cfg.padding_value if self.padding == 'constant_value' else None
        self.frame_drop_rate = cfg.frame_drop_rate

    def check_dominant_hand(self, array):
        '''
            non-dominant hand's array is all nan

            Input:
                array: (seq_len, 543, 2)
            Returns:
                array: (seq_len, 20, 2)
        '''

        right_hand = array[:, self.array_dict['right_hand'], :]
        left_hand = array[:, self.array_dict['left_hand'], :]
        right_nan_length = np.isnan(right_hand[:, 0, 0]).sum()
        left_nan_length = np.isnan(left_hand[:, 0, 0]).sum()
        return 'left' if right_nan_length > left_nan_length else 'right'

    def array_process(self, array, landmark, max_length):
        '''
            - delete nan frame and to tensor
            - pad or truncate
            - to tensor

            Input:
                array: (seq_len, n_landmarks, 2)
            Returns:
                array: (max_length, n_landmarks, 2)
        '''
        array = array[:, self.array_dict[landmark], :].copy()
        n_landmarks = len(self.array_dict[landmark])

        # dropna
        not_nan_frame = ~np.isnan(np.mean(array, axis=(1, 2)))
        array = array[not_nan_frame, :, :]
        # pad and truncate
        if len(array) < max_length:
            # pad
            pad_length = max_length - len(array)
            if self.padding == 'edge':
                array = np.pad(
                    array, ((0, pad_length), (0, 0), (0, 0)), 'edge')
            elif self.padding == 'constant_value':
                array = np.pad(array, ((0, pad_length), (0, 0), (0, 0)),
                               'constant', constant_values=self.padding_value)
        else:
            # truncate
            array = array[:max_length]
        # dim (1, 2) -> 1
        array = array.reshape(max_length, n_landmarks * 2)
        # to tensor
        tensor = torch.from_numpy(array)
        return tensor

    def mirrored(self, array):
        '''
            Process:
                1. (right/leftが分かれている => 反転)
                2. x座標を反転
            Returns:
                mirrered_array: (seq_len, 543, 2)
        '''
        def invert_x(tmp):
            tmp = tmp.copy()
            tmp[:, :, 0] = -tmp[:, :, 0]
            return tmp
        mirrered_array = array.copy()
        # hand
        mirrered_array[:, self.array_dict['right_hand'], :] = invert_x(
            array[:, self.array_dict['left_hand'], :])
        mirrered_array[:, self.array_dict['left_hand'], :] = invert_x(
            array[:, self.array_dict['right_hand'], :])
        # lips
        for key in ['lips']:
            mirrered_array[:, self.array_dict[key], :] = invert_x(
                array[:, self.array_dict[key], :])
        return mirrered_array

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
            Returns:
                array: (seq_len, 543, 2)
                label: (seq_len)
        '''
        row = self.df.iloc[idx]
        lmdb_id = int(row['lmdb_id'])
        with self.env.begin(write=False) as txn:
            label_key = f'label-{str(lmdb_id).zfill(8)}'.encode()
            label = txn.get(label_key).decode('utf-8')
            array_key = f'array-{str(lmdb_id).zfill(8)}'.encode()
            array = np.frombuffer(txn.get(array_key),
                                  dtype=np.float16).reshape(-1, 543, 2).copy()
            # -100 -> nan
            array[array == -100] = np.nan

        # check dominant hand
        dominant_hand = self.check_dominant_hand(array)
        if dominant_hand == 'left':
            array = self.mirrored(array)

        # hand array
        hand_tensor = self.array_process(
            array, 'right_hand', self.hand_max_length)

        # lips array
        lips_tensor = self.array_process(
            array, 'lips', self.lips_max_length)

        # label to token and to tensor
        item = {
            'hand': hand_tensor,
            'lips': lips_tensor,
            'label': label
        }
        return item


# def collate_fn(batch):
#     '''
#         Returns:
#             hand: (bs, seq_len, 20, 2)
#             lips: (bs, seq_len, 11, 2)
#             label: (bs, seq_len)
#     '''
#     hand = [item['hand'] for item in batch]
#     lips = [item['lips'] for item in batch]
#     label = [item['label'] for item in batch]

#     hand = torch.stack(hand, dim=0)
#     lips = torch.stack(lips, dim=0)
#     label = torch.stack(label, dim=0)
#     return hand, lips, label

# prepare dataloader


def prepare_dataloader(cfg, LMDB_DIR, char_to_idx, use_landmarks, train_fold_df, valid_fold_df):
    train_dataset = Asl2Dataset(
        cfg, train_fold_df, LMDB_DIR, char_to_idx, use_landmarks)
    valid_dataset = Asl2Dataset(
        cfg, valid_fold_df, LMDB_DIR, char_to_idx, use_landmarks)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.valid_bs, shuffle=False, num_workers=os.cpu_count())
    return train_loader, valid_loader

# model


class Model(nn.Module):
    def __init__(self, seq_len, n_features, n_class):
        super(Model, self).__init__()
        self.bn0 = nn.BatchNorm1d(n_features)  # 各landmarkでmean,std=0,1にする
        self.conv1 = nn.Conv1d(n_features, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, n_class)

    def forward(self, x):
        '''
            input: (bs, 42, 576)
            output: (bs, 576(seq_len), 59(n_classes))
        '''
        x = self.bn0(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))  # (bs, 42, 576)
        h = h.permute(0, 2, 1)  # (bs, 576, 42)
        h = self.fc(h)  # (bs, 42, 59)
        return h


def create_model(cfg, use_landmarks, n_classes):
    # model = ModelFixedLen(
    #     in_dim_H=cfg.hand_max_length,
    #     in_dim_L=cfg.lips_max_length,
    #     dim1_H=cfg.dim1_H,
    #     dim1_L=cfg.dim1_L,
    #     dim2=cfg.dim2,
    #     n_class=n_classes,
    # )
    n_features = len(use_landmarks['right_hand']) * 2
    model = Model(cfg.hand_max_length, n_features, n_classes)
    return model


# train


def train_function(
    cfg,
    fold,
    epoch,
    train_loader,
    ctc_converter,
    model,
    optimizer,
    scheduler,
    loss_fn,
    scaler,
    device,
):
    model.train()
    train_loss = AverageMeter()
    train_norm_ld = AverageMeter()
    train_accuracy = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for _, batch in pbar:
        bs = len(batch.shape[0])

        hand = batch['hand'].to(device).float()
        # lips = batch['lips'].to(device)
        label_tensor, len_label_tensor = ctc_converter.encode(
            batch['label'], batch_max_length=cfg.batch_max_length).to(device)
        label_tensor = label_tensor.to(device)
        len_label_tensor = len_label_tensor.to(device)

        with autocast():
            preds = model(hand)  # (bs, seq_len, n_classes)
            preds = preds.log_softmax(2).permute(
                1, 0, 2)  # (seq_len, bs, n_classes)
            preds_size = torch.IntTensor([preds.size(1)] * bs)
            loss = loss_fn(preds, label_tensor, preds_size, len_label_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        # to numpy
        preds = preds.argmax(dim=-1).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        len_label = len_label_tensor.detach().cpu().numpy()

        pred_text, label_text = ctc_converter.decode(
            preds), ctc_converter.decode(label, len_label)

        accuracy, norm_ld = validation_metrics(pred_text, label_text)

        train_loss.update(loss.item(), bs)
        train_norm_ld.update(norm_ld, bs)
        train_accuracy.update(accuracy, bs)

        # pbar
        pbar.set_description(
            f'fold: {fold}, epoch: {epoch}, loss: {train_loss.avg:.4f}, norm_ld: {train_norm_ld.avg:.4f}, accuracy: {train_accuracy.avg:.4f}, lr: {get_lr(optimizer):.4f}')

    return train_loss.avg, train_norm_ld.avg, train_accuracy.avg


def valid_function(
        cfg,
        fold,
        epoch,
        valid_loader,
        ctc_converter,
        model,
        loss_fn,
        device
):
    model.eval()
    valid_loss = AverageMeter()
    valid_norm_ld = AverageMeter()
    valid_accuracy = AverageMeter()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for _, batch in pbar:
        bs = len(batch.shape[0])

        hand = batch['hand'].to(device).float()
        # lips = batch['lips'].to(device)
        label_tensor, len_label_tensor = ctc_converter.encode(
            batch['label'], batch_max_length=cfg.batch_max_length).to(device)
        label_tensor = label_tensor.to(device)
        len_label_tensor = len_label_tensor.to(device)

        with torch.no_grad():
            preds = model(hand)  # (bs, seq_len, n_classes)
            preds = preds.log_softmax(2).permute(
                1, 0, 2)  # (seq_len, bs, n_classes)
            preds_size = torch.IntTensor([preds.size(1)] * bs)
            loss = loss_fn(preds, label_tensor, preds_size, len_label_tensor)

        # to numpy
        preds = preds.argmax(dim=-1).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        len_label = len_label_tensor.detach().cpu().numpy()

        pred_text, label_text = ctc_converter.decode(
            preds), ctc_converter.decode(label, len_label)

        accuracy, norm_ld = validation_metrics(pred_text, label_text)

        valid_loss.update(loss.item(), bs)
        valid_norm_ld.update(norm_ld, bs)
        valid_accuracy.update(accuracy, bs)

        # pbar
        pbar.set_description(
            f'fold: {fold}, epoch: {epoch}, loss: {valid_loss.avg:.4f}, norm_ld: {valid_norm_ld.avg:.4f}, accuracy: {valid_accuracy.avg:.4f}')

    return valid_loss.avg, valid_norm_ld.avg, valid_accuracy.avg


# main
def main():
    EXP_PATH = Path.cwd()
    with initialize_config_dir(config_dir=str(EXP_PATH / 'config')):
        cfg = compose(config_name='config.yaml')

    ROOT_DIR = Path.cwd().parents[2]
    exp_name = EXP_PATH.name
    RAW_DATA_DIR = ROOT_DIR / 'data' / 'original_data'
    DATA_DIR = ROOT_DIR / 'data' / 'created_data' / cfg.dataset_name
    LMDB_DIR = DATA_DIR / 'train' / 'lmdb'
    TRAIN_CSV_PATH = DATA_DIR / 'train2.csv'
    SAVE_DIR = ROOT_DIR / 'outputs' / exp_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # seed
    seed_everything(cfg.seed)

    # wandb
    wandb.login()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_df = split_data(cfg, TRAIN_CSV_PATH)
    char_to_idx = json.load(
        open(RAW_DATA_DIR / 'character_to_prediction_index.json'))
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    # use landmark index
    use_landmarks = get_indices(cfg)

    for fold in range(cfg.n_folds):
        # wandb init
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=cfg.wandb_project,
            entity='luka-magic',
            name=f'{exp_name}',
            config=wandb.config,
            mode='online' if cfg.use_wandb else 'disabled'
        )
        wandb.config.fold = fold

        # fold df
        train_fold_df = train_df[train_df['fold']
                                 != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df['fold']
                                 == fold].reset_index(drop=True)
        train_loader, valid_loader = prepare_dataloader(
            cfg, LMDB_DIR, char_to_idx, use_landmarks, train_fold_df, valid_fold_df)

        # ctc converter
        ctc_converter = CTCLabelConverter(char_to_idx)
        n_chars = len(ctc_converter.character)

        # model
        model = create_model(
            cfg, use_landmarks, n_classes=n_chars)

        # optimizer
        if cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # scheduler
        if cfg.scheduler == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        else:
            scheduler = None

        # loss
        loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # scaler
        scaler = GradScaler()

        # train
        best_score = {
            'loss': np.inf,
            'levenshtein': 0,
            'accuracy': 0,
        }
        for epoch in range(1, cfg.n_epochs+1):
            train_loss, train_norm_ld, train_accuracy = train_function(
                cfg, fold, epoch, train_loader, ctc_converter, model, optimizer, scheduler, loss_fn, scaler, device)
            valid_loss, valid_norm_ld, valid_accuracy = valid_function(
                cfg, fold, epoch, valid_loader, ctc_converter, model, loss_fn, device)

            # wandb log
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_norm_ld': train_norm_ld,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_norm_ld': valid_norm_ld,
                'valid_accuracy': valid_accuracy,
            })

            if valid_norm_ld > best_score['levenshtein']:
                best_score['loss'] = valid_loss
                best_score['levenshtein'] = valid_norm_ld
                best_score['accuracy'] = valid_accuracy
                # save model
                torch.save(model.state_dict(), SAVE_DIR /
                           f'best_levenshtein_fold{fold}.pth')
                wandb.run.summary['best_levenshtein'] = best_score['levenshtein']


if __name__ == '__main__':
    main()
