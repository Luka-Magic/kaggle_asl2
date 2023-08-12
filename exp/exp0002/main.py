# basic
import sys
import gc
import math
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import random
from tqdm import tqdm
from collections import OrderedDict, Counter
import lmdb
# from typing import List, Dict, Union, Tuple, Any

# hydra
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

from utils import seed_everything, AverageMeter, get_lr, validation_metrics, LabelConverter
from augment import AffineMatTools
from transformers import Transformer

PAD_TOKEN = 'P'
SOS_TOKEN = 'S'
EOS_TOKEN = 'E'


def init_lmdb(lmdb_dir):
    '''
        ただただ1回lmdbのデータを全て取り出す。
        これをしないとdataloaderでとんでもない時間がかかる。
        (この処理は約10分、そのマシンで初めて実行するときだけこの関数を実行する(argsに指定する))
    '''
    assert lmdb_dir.exists(), f'{lmdb_dir} does not exist'
    env = lmdb.open(str(lmdb_dir), max_readers=32,
                    readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()).decode('utf-8'))

    with env.begin(write=False) as txn:
        for lmdb_id in tqdm(range(n_samples)):
            lmdb_id = int(lmdb_id)
            label_key = f'label-{str(lmdb_id+1).zfill(8)}'.encode()
            _ = txn.get(label_key).decode('utf-8')
            array_key = f'array-{str(lmdb_id+1).zfill(8)}'.encode()
            _ = np.frombuffer(txn.get(array_key), dtype=np.float16)


def split_data(cfg, train_csv_path):
    '''
        StratifiedGroupKFold
            group: participant_id
            target: phrase_type (url, telephone, address)
    '''

    train_df = pd.read_csv(train_csv_path)

    # filtering
    train_df['phrase_length'] = train_df['phrase'].map(len)
    train_df['n_frames_hand_per_char'] = train_df['n_frames_hand'] / \
        train_df['phrase_length']
    raw_len = len(train_df)
    train_df = train_df.query('n_frames_hand_per_char > 3.0').reset_index()
    print(
        f'filtered {raw_len - len(train_df)} samples, {len(train_df)} samples left')

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

        left_pose=[502, 504, 506, 508, 510],
        right_pose=[503, 505, 507, 509, 511],
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
    def __init__(self, cfg, df, lmdb_dir, converter, use_landmarks):
        self.cfg = cfg
        self.df = df
        self.env = lmdb.open(str(lmdb_dir), max_readers=32,
                             readonly=True, lock=False, readahead=False, meminit=False)
        self.array_dict = use_landmarks
        self.converter = converter

        self.hand_max_length = cfg.hand_max_length
        self.lips_max_length = cfg.lips_max_length
        self.pose_max_length = cfg.pose_max_length

        self.phrase_max_length = cfg.phrase_max_length
        self.max_length = max(self.hand_max_length, self.phrase_max_length)

        self.padding = cfg.padding
        self.padding_value = cfg.padding_value if self.padding == 'constant_value' else None
        self.frame_drop_rate = cfg.frame_drop_rate
        self.aug_hand_params = cfg.aug_hand_params

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

    def apply_aug_right_hand(self, hand, debug=False):
        angle = random.gauss(0, self.aug_hand_params["angle"] / 2)
        scale = random.gauss(1, self.aug_hand_params["scale"] / 2)
        shift_x = random.gauss(0, self.aug_hand_params["shift_x"] / 2)
        shift_y = random.gauss(0, self.aug_hand_params["shift_y"] / 2)

        amt = AffineMatTools()
        amt.scale(scale)
        amt.rotation_degree(angle)
        amt.shift(shift_x, shift_y)

        aug_hand = hand - hand[:, 0][:, None]

        # aug_hand_z = aug_hand[:, :, 2][:, :, None]
        # aug_hand = aug_hand[:, :, :2]
        aug_hand = amt.transform(aug_hand)
        # aug_hand = np.concatenate((aug_hand, aug_hand_z), axis=2)
        aug_hand = aug_hand + hand[:, 0][:, None]
        aug_hand = aug_hand.astype(np.float32)
        return aug_hand

    def array_process(self, array, landmark, max_length):
        '''
        - slice landmark array
        - if right_hand => apply_aug_hand
        - pad or truncate
        - to tensor

        Parameters
        ----------
        array: np.array
            shape: (seq_len, n_landmarks, 2)

        Returns
        -------
        tensor: torch.tensor
            shape: (max_length, n_landmarks, 2)
        tensor_length: torch.tensor
            shape: (1)
        '''

        # slice
        array = array[:, self.array_dict[landmark], :].copy()
        n_landmarks = len(self.array_dict[landmark])

        # apply aug
        if landmark == 'right_hand':
            array = self.apply_aug_right_hand(array)

        # normalization x and y
        for i in range(2):
            array_1d = array[:, :, i].reshape(-1)
            array[:, :, i] = (
                array[:, :, i] - np.nanmean(array_1d)) / np.nanstd(array_1d)

        # no frame
        if len(array) == 0:
            array = np.zeros((max_length, n_landmarks, 2))

        # landmark length
        landmark_length = min(len(array), max_length)

        # pad or truncate
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
        tensor = torch.from_numpy(array)  # (seq_len, input_size)

        return tensor, landmark_length

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

    def create_mask(self, landmark_length, label_length):
        NEG_INFTY = -1e9

        # Creates a tensor with all values = True
        look_ahead_mask = torch.full([self.max_length, self.max_length], True)
        # Upper traingle = True only
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        # print(look_ahead_mask)
        encoder_padding_mask = torch.full(
            [self.max_length, self.max_length], False)
        decoder_padding_mask_self_attention = torch.full(
            [self.max_length, self.max_length], False)
        decoder_padding_mask_cross_attention = torch.full(
            [self.max_length, self.max_length], False)
        # print(encoder_padding_mask)

        frame_chars_to_padding_mask = np.arange(
            landmark_length + 1, self.max_length)
        eng_chars_to_padding_mask = np.arange(
            label_length + 1, self.max_length)

        encoder_padding_mask[:, frame_chars_to_padding_mask] = True
        encoder_padding_mask[frame_chars_to_padding_mask, :] = True

        decoder_padding_mask_self_attention[:,
                                            eng_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[eng_chars_to_padding_mask, :] = True

        decoder_padding_mask_cross_attention[:,
                                             eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[frame_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(
            encoder_padding_mask, NEG_INFTY, 0)
        decoder_self_attention_mask = torch.where(
            look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
        decoder_cross_attention_mask = torch.where(
            decoder_padding_mask_cross_attention, NEG_INFTY, 0)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

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

        # drop nan
        hand_not_nan_frame = ~np.isnan(
            np.mean(array[:, self.array_dict['right_hand'], :], axis=(1, 2)))
        array = array[hand_not_nan_frame, :, :]

        # hand array
        hand_tensor, hand_length = self.array_process(
            array, 'right_hand', self.hand_max_length)

        # lips array
        lips_tensor, _ = self.array_process(
            array, 'lips', self.lips_max_length)

        # pose array
        right_pose_tensor, _ = self.array_process(
            array, 'right_pose', self.pose_max_length)
        left_pose_tensor, _ = self.array_process(
            array, 'left_pose', self.pose_max_length)

        # concat
        # input_tensor = torch.cat(
        #     (hand_tensor, lips_tensor, right_pose_tensor, left_pose_tensor), dim=1)
        input_tensor = hand_tensor
        # label to token and to tensor
        input_label_tensor, label_length = self.converter.encode(
            label, add_sos=True)
        target_tensor, _ = self.converter.encode(
            label, add_sos=False)
        # create mask
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(
            hand_length, label_length)
        item = {
            'input': input_tensor,
            'input_label': input_label_tensor,
            'target': target_tensor,
            'target_length': len(label),
            'enc_self_attn_msk': encoder_self_attention_mask,
            'dec_self_attn_msk': decoder_self_attention_mask,
            'dec_cross_attn_msk': decoder_cross_attention_mask,
        }
        return item

# collate_fn


# def collate_fn(batch):
#     # pad label
#     max_len = max([len(item['label']) for item in batch])
#     for item in batch:
#         pad_len = max_len - len(item['label'])
#         item['label'] = item['label'] + [0] * pad_len

def prepare_dataloader(cfg, LMDB_DIR, converter, use_landmarks, train_fold_df, valid_fold_df):
    train_dataset = Asl2Dataset(
        cfg, train_fold_df, LMDB_DIR, converter, use_landmarks)
    valid_dataset = Asl2Dataset(
        cfg, valid_fold_df, LMDB_DIR, converter, use_landmarks)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train_bs, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.valid_bs, shuffle=False, num_workers=os.cpu_count())
    return train_loader, valid_loader

# model


def create_model(cfg, input_size, vocab_size, max_seq_length):
    model = Transformer(
        input_size=input_size,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embed_dim=cfg.embed_dim,
        ffn_hidden=cfg.ffn_hidden,
        num_heads=cfg.num_heads,
        drop_prob=cfg.drop_prob,
        num_layers=cfg.num_layers,
    )
    return model


# train
def train_function(
    cfg,
    fold,
    epoch,
    train_loader,
    converter,
    model,
    optimizer,
    scheduler,
    scheduler_step_frequence,
    loss_fn,
    scaler,
    device,
):
    model.train()
    train_loss = AverageMeter()
    train_norm_ld = AverageMeter()
    train_accuracy = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in pbar:
        bs = len(batch['target'])

        hand = batch['input'].to(device).float()
        # lips = batch['lips'].to(device)
        input_label = batch['input_label'].to(device)
        targets = batch['target'].to(device)
        target_length = batch['target_length'].to(device)
        enc_self_attn_msk = batch['enc_self_attn_msk'].to(device)
        dec_self_attn_msk = batch['dec_self_attn_msk'].to(device)
        dec_cross_attn_msk = batch['dec_cross_attn_msk'].to(device)
        with autocast():
            # hand: shape=(bs, seq_len, input_size)
            # input_label: shape=(bs, label_len)
            # mask: shape=(bs, seq_len, seq_len)
            preds = model(hand, input_label, enc_self_attn_msk, dec_self_attn_msk,
                          dec_cross_attn_msk)
            # preds: shape=(bs, seq_len, vocab_size)
            loss = loss_fn(preds.transpose(-1, -2), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None and scheduler_step_frequence == 'step':
            scheduler.step()

        # to numpy
        preds = preds.argmax(
            dim=-1).detach().cpu().numpy()  # (bs, seq_len)
        targets = targets.detach().cpu().numpy()
        target_length = target_length.detach().cpu().numpy()  # (bs, label_len)

        preds_text = [converter.decode(pred) for pred in preds]
        targets_text = [converter.decode(target) for target in targets]

        if i < 5:
            print([(pred_, label_)
                  for pred_, label_ in zip(preds_text, targets_text)])

        accuracy, norm_ld = validation_metrics(preds_text, targets_text)

        train_loss.update(loss.item(), bs)
        train_norm_ld.update(norm_ld, bs)
        train_accuracy.update(accuracy, bs)

        # pbar
        pbar.set_description(f'【TRAIN EPOCH {epoch}/{cfg.n_epochs}】')
        pbar.set_postfix(OrderedDict(loss=train_loss.avg, norm_ld=train_norm_ld.avg,
                         accuracy=train_accuracy.avg, lr=get_lr(optimizer)))
    if scheduler is not None and scheduler_step_frequence == 'epoch':
        scheduler.step()
    return train_loss.avg, train_norm_ld.avg, train_accuracy.avg


# valid
def valid_function(
    cfg,
    fold,
    epoch,
    valid_loader,
    converter,
    model,
    loss_fn,
    device,
):
    model.eval()
    valid_loss = AverageMeter()
    valid_norm_ld = AverageMeter()
    valid_accuracy = AverageMeter()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for i, batch in pbar:
        bs = len(batch['target'])

        hand = batch['input'].to(device).float()
        # lips = batch['lips'].to(device)
        input_label = batch['input_label'].to(device)
        targets = batch['target'].to(device)
        target_length = batch['target_length'].to(device)
        enc_self_attn_msk = batch['enc_self_attn_msk'].to(device)
        dec_self_attn_msk = batch['dec_self_attn_msk'].to(device)
        dec_cross_attn_msk = batch['dec_cross_attn_msk'].to(device)
        with torch.no_grad():
            # hand: shape=(bs, seq_len, input_size)
            # input_label: shape=(bs, label_len)
            # mask: shape=(bs, seq_len, seq_len)
            preds = model(hand, input_label, enc_self_attn_msk, dec_self_attn_msk,
                          dec_cross_attn_msk)
            # preds: shape=(bs, seq_len, vocab_size)
            loss = loss_fn(preds.transpose(-1, -2), targets)

        # to numpy
        preds = preds.argmax(
            dim=-1).detach().cpu().numpy()  # (bs, seq_len)
        targets = targets.detach().cpu().numpy()
        target_length = target_length.detach().cpu().numpy()  # (bs, label_len)

        preds_text = [converter.decode(pred) for pred in preds]
        targets_text = [converter.decode(target) for target in targets]

        if i < 5:
            print([(pred_, label_)
                  for pred_, label_ in zip(preds_text, targets_text)])

        accuracy, norm_ld = validation_metrics(preds_text, targets_text)

        valid_loss.update(loss.item(), bs)
        valid_norm_ld.update(norm_ld, bs)
        valid_accuracy.update(accuracy, bs)

        # pbar
        pbar.set_description(f'【VALID EPOCH {epoch}/{cfg.n_epochs}】')
        pbar.set_postfix(OrderedDict(loss=valid_loss.avg, norm_ld=valid_norm_ld.avg,
                         accuracy=valid_accuracy.avg))
    return valid_loss.avg, valid_norm_ld.avg, valid_accuracy.avg
# main


def main(
    is_first_learning,
    use_wandb
):
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

    # wandb
    wandb.login()

    # lmdb init
    # はじめに読み込みを行わないとdataloaderでとんでもない時間がかかる
    if is_first_learning:
        init_lmdb(LMDB_DIR)

    # seed
    seed_everything(cfg.seed)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_df = split_data(cfg, TRAIN_CSV_PATH)
    char_to_idx = json.load(
        open(RAW_DATA_DIR / 'character_to_prediction_index.json'))
    # use landmark index
    use_landmarks = get_indices(cfg)

    # max length
    max_length = max(cfg.hand_max_length, cfg.phrase_max_length)

    for fold in range(cfg.n_folds):
        if fold not in cfg.use_fold:
            continue

        # wandb init
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=cfg.wandb_project,
            entity='luka-magic',
            name=f'{exp_name}',
            config=wandb.config,
            mode='online' if use_wandb == 1 else 'disabled'
        )
        wandb.config.fold = fold

        # converter
        converter = LabelConverter(
            char_to_idx, max_length, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)
        vocab_size = len(converter.character)

        # fold df
        train_fold_df = train_df[train_df['fold']
                                 != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df['fold']
                                 == fold].reset_index(drop=True)
        train_loader, valid_loader = prepare_dataloader(
            cfg, LMDB_DIR, converter, use_landmarks, train_fold_df, valid_fold_df)

        # model
        model = create_model(
            cfg, input_size=42, vocab_size=vocab_size, max_seq_length=max_length).to(device)

        # optimizer
        if cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # scheduler
        if cfg.scheduler == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, total_steps=cfg.n_epochs * len(train_loader), max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
            scheduler_step_frequence = cfg.scheduler_step_frequence
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
            scheduler_step_frequence = cfg.scheduler_step_frequence
        else:
            scheduler = None
            scheduler_step_frequence = None

        # loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

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
                cfg, fold, epoch, train_loader, converter, model, optimizer, scheduler, scheduler_step_frequence, loss_fn, scaler, device)
            valid_loss, valid_norm_ld, valid_accuracy = valid_function(
                cfg, fold, epoch, valid_loader, converter, model, loss_fn, device)

            print('-*-'*30)
            print(f'【FOLD {fold} EPOCH {epoch}/{cfg.n_epochs}】')
            print('    train_loss: {:.4f}'.format(train_loss))
            print('    train_norm_ld: {:.4f}'.format(train_norm_ld))
            print('    train_accuracy: {:.4f}'.format(train_accuracy))
            print('    valid_loss: {:.4f}'.format(valid_loss))
            print('    valid_norm_ld: {:.4f}'.format(valid_norm_ld))
            print('    valid_accuracy: {:.4f}'.format(valid_accuracy))
            print('-*-'*30)

            # wandb log
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_norm_ld': train_norm_ld,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_norm_ld': valid_norm_ld,
                'valid_accuracy': valid_accuracy,
                'lr': get_lr(optimizer)
            })

            if valid_norm_ld > best_score['levenshtein']:
                best_score['loss'] = valid_loss
                best_score['levenshtein'] = valid_norm_ld
                best_score['accuracy'] = valid_accuracy
                # save model
                torch.save(model.state_dict(), SAVE_DIR /
                           f'best_levenshtein_fold{fold}.pth')
                wandb.run.summary['best_levenshtein'] = best_score['levenshtein']
        wandb.finish()


if __name__ == '__main__':
    is_first_learning = int(sys.argv[1])
    use_wandb = int(sys.argv[2])
    main(is_first_learning, use_wandb)
