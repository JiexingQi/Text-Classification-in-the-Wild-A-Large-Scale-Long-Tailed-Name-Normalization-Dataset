import os
import pickle
from collections import OrderedDict

import torch

import cs
from . import config
from tools.tokenizer import AnnTokenizer
from .dataset import BertMetricDataset, BertMetricTestSet
from models.ann_bert import AnnBert


def get_model(
        *,
        pretrained_model=None,
        last_training_time=None,
        last_step=None,
):
    if pretrained_model is None:
        pretrained_model = config.pretrained_model
    if last_training_time is None:
        last_training_time = config.last_training_time
    if last_step is None:
        last_step = config.last_step

    if last_training_time == 0:
        conf = config.model_conf
    else:
        conf_file = os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{last_training_time}', 'model_conf.pkl')
        with open(conf_file, 'rb') as f:
            conf = pickle.load(f)

    model = AnnBert(conf)

    if last_training_time != 0:
        file = os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{last_training_time}', f'model_{last_step}.pt')
        model.load_state_dict(torch.load(file, map_location=torch.device("cpu")), strict=False)
    elif pretrained_model is not None:
        file = os.path.join(cs.SAVED_MODEL_DIR, pretrained_model)
        state = torch.load(file, map_location=torch.device("cpu"))
        new_state = OrderedDict()
        for k, v in state.items():
            if k.startswith('bert.'):
                new_state[k[5:]] = v
        model.load_state_dict(new_state, strict=False)

    return model


def get_training_set():
    tokenizer = AnnTokenizer(cs.VOCAB_FILE)
    dataset = BertMetricDataset(
        tokenizer,
        manager_host=config.manager_host,
        manager_port=config.manager_port,
        manager_password=config.manager_password,
        batch_size=config.batch_size,
        samples_per_class=config.samples_per_class,
        resample_exp=config.resample_exp,
    )
    return dataset


def get_test_set(part='test'):
    assert part in ['validation', 'test', 'anchor']
    if part == 'test':
        dataset_file = config.test_set_file
    elif part == 'validation':
        dataset_file = config.val_set_file
    else:
        dataset_file = config.anchors_file

#     tokenizer = AnnTokenizer(cs.VOCAB_FILE)
#     with open(dataset_file, 'rb') as f:
#         dataset = pickle.load(f)
#     for i, (name, aff_id) in enumerate(dataset):
#         dataset[i] = (name, int(aff_id))
        
    tokenizer = AnnTokenizer(cs.VOCAB_FILE)
    with open(config.id_to_cls_file, 'rb') as f:
        id_to_cls = pickle.load(f)
    with open(dataset_file, 'rb') as f:
        raw_dataset = pickle.load(f)
    dataset = []
    for aff_name, aff_id in raw_dataset:
        dataset.append((aff_name, id_to_cls[aff_id]))
        
    test_set = BertMetricTestSet(
        tokenizer,
        dataset,
    )
    return test_set



def get_osc_set(dataset_file):
    tokenizer = AnnTokenizer(cs.VOCAB_FILE)
    with open(config.id_to_cls_file, 'rb') as f:
        id_to_cls = pickle.load(f)
        
    nor2afid = pickle.load(open(cs.save_pkl_root+"nor2afid.pkl", "rb"))
    
    texts = []
    labels = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            ori, nor, open_label = line.replace("\n", "").split("\t\t")
            afid = nor2afid[nor]
            texts.append(ori)
            labels.append(afid)
    dataset = []
    for aff_name, aff_id in zip(texts, labels):
        dataset.append((aff_name, id_to_cls.get(aff_id, -1)))
    test_set = BertMetricTestSet(
        tokenizer,
        dataset,
    )
    return test_set


def get_osv_set(dataset_file):
    tokenizer = AnnTokenizer(cs.VOCAB_FILE)
    
    texts_first = []
    texts_second = []
    labels = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            ori_first, ori_second, ver_label = line.replace("\n", "").split("\t\t")
            texts_first.append(ori_first)
            texts_second.append(ori_second)
            labels.append(int(ver_label))
    dataset_first, dataset_second = [], []
    for aff_name_first, aff_name_second, aff_id in zip(texts_first, texts_second, labels):
        dataset_first.append((aff_name_first, aff_id))
        dataset_second.append((aff_name_second, aff_id))
    test_set_first = BertMetricTestSet(
        tokenizer,
        dataset_first,
    )
    test_set_second = BertMetricTestSet(
        tokenizer,
        dataset_second,
    )
    return (test_set_first, test_set_second)