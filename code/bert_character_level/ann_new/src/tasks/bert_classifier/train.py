import itertools
import os
import pickle
import time
import math

import torch
from torch.utils.tensorboard import SummaryWriter 
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_constant_schedule_with_warmup

import cs
from . import config
from tools.logger import Logger
from tools.utils import save_model, log_config, to_device, get_training_set_loader, StopTraining, only_main_process
from .predict import bert_classifier_validation
from .utils import get_model, get_training_set

logger: Logger
start_time = int(time.time())
last_log_time = start_time
last_save_time = start_time
loss_sum = 0
loss_cnt = 0
batch_cnt = 0


def generate_data(dataloader, sampler):
    global logger, batch_cnt
    for epoch in itertools.count():
        logger.info(f'epoch: {epoch} begin.')
        if sampler is not None:
            sampler.set_epoch(epoch)
        for data in dataloader:
            yield data
            batch_cnt += 1
        logger.info(f'epoch: {epoch} end.')


def train_one_batch(model, loss_fn, opt, data):
    model.train()
    opt.zero_grad()

    char_ids, mask, char_position_ids, word_position_ids, labels = [d.to(cs.device) for d in data]
    output = model(
        char_ids,
        attention_mask=mask,
        char_position_ids=char_position_ids,
        word_position_ids=word_position_ids,
    )
    logits = output[0]
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()

    return loss.item()


def after_batch(model, opt, loss, scheduler):
    global loss_sum, loss_cnt
    step_no = scheduler.last_epoch
    batch_cnt_in_step = batch_cnt % config.batches_per_step

    if cs.rank == 0:
        loss_sum += loss
        loss_cnt += 1
        print('\r', step_no, batch_cnt_in_step,
              f'{(loss_sum / loss_cnt):.5f}     ', end='')

    if batch_cnt_in_step == config.batches_per_step - 1:
        after_step(step_no, model, opt)
        scheduler.step()


@only_main_process
def after_step(step_no, model, opt):
    
    step = step_no
    global loss_sum, loss_cnt, last_log_time, last_save_time
    if time.time() - last_log_time > config.log_interval:
        logger.info(f'step_no: {step_no}, avg_loss: {(loss_sum / loss_cnt):.7f}, '
                    f'current lr: {opt.param_groups[0]["lr"]}')
        writer = SummaryWriter('./tensorboard')
        writer.add_scalar('train/avg_loss', (loss_sum / loss_cnt), step)
        writer.add_scalar('train/lr', opt.param_groups[0]["lr"], step)
        val_result, val_value = bert_classifier_validation(model)
        test_result, test_value = bert_classifier_validation(model, test_set='test_set')
        logger.info(f'step_no: {step_no}, val result: {val_result}')
        logger.info(f'step_no: {step_no}, test result: {test_result}')

        dev_overall_acc, dev_overall_p, dev_overall_r, dev_overall_f1 = val_value[0], val_value[1], val_value[2], val_value[3]
        dev_high_acc, dev_high_p, dev_high_r, dev_high_f1 = val_value[4], val_value[5], val_value[6], val_value[7]
        dev_middle_acc, dev_middle_p, dev_middle_r, dev_middle_f1 = val_value[8], val_value[9], val_value[10], val_value[11]
        dev_few_acc, dev_few_p, dev_few_r, dev_few_f1 = val_value[12], val_value[13], val_value[14], val_value[15]

        writer.add_scalar('overall/dev_overall_acc', dev_overall_acc, step)
        writer.add_scalar('overall/dev_overall_p', dev_overall_p, step)
        writer.add_scalar('overall/dev_overall_r', dev_overall_r, step)
        writer.add_scalar('overall/dev_overall_f1', dev_overall_f1, step)
        writer.add_scalar('high/dev_high_acc', dev_high_acc, step)
        writer.add_scalar('high/dev_high_p', dev_high_p, step)
        writer.add_scalar('high/dev_high_r', dev_high_r, step)
        writer.add_scalar('high/dev_high_f1', dev_high_f1, step)
        writer.add_scalar('middle/dev_middle_acc', dev_middle_acc, step)
        writer.add_scalar('middle/dev_middle_p', dev_middle_p, step)
        writer.add_scalar('middle/dev_middle_r', dev_middle_r, step)
        writer.add_scalar('middle/dev_middle_f1', dev_middle_f1, step)
        writer.add_scalar('few/dev_few_acc', dev_few_acc, step)
        writer.add_scalar('few/dev_few_p', dev_few_p, step)
        writer.add_scalar('few/dev_few_r', dev_few_r, step)
        writer.add_scalar('few/dev_few_f1', dev_few_f1, step)

        test_overall_acc, test_overall_p, test_overall_r, test_overall_f1 = test_value[0], test_value[1], test_value[2], test_value[3]
        test_high_acc, test_high_p, test_high_r, test_high_f1 = test_value[4], test_value[5], test_value[6], test_value[7]
        test_middle_acc, test_middle_p, test_middle_r, test_middle_f1 = test_value[8], test_value[9], test_value[10], test_value[11]
        test_few_acc, test_few_p, test_few_r, test_few_f1 = test_value[12], test_value[13], test_value[14], test_value[15] 

        writer.add_scalar('overall/test_overall_acc', test_overall_acc, step)
        writer.add_scalar('overall/test_overall_p', test_overall_p, step)
        writer.add_scalar('overall/test_overall_r', test_overall_r, step)
        writer.add_scalar('overall/test_overall_f1', test_overall_f1, step)
        writer.add_scalar('high/test_high_acc', test_high_acc, step)
        writer.add_scalar('high/test_high_p', test_high_p, step)
        writer.add_scalar('high/test_high_r', test_high_r, step)
        writer.add_scalar('high/test_high_f1', test_high_f1, step)
        writer.add_scalar('middle/test_middle_acc', test_middle_acc, step)
        writer.add_scalar('middle/test_middle_p', test_middle_p, step)
        writer.add_scalar('middle/test_middle_r', test_middle_r, step)
        writer.add_scalar('middle/test_middle_f1', test_middle_f1, step)
        writer.add_scalar('few/test_few_acc', test_few_acc, step)
        writer.add_scalar('few/test_few_p', test_few_p, step)
        writer.add_scalar('few/test_few_r', test_few_r, step)
        writer.add_scalar('few/test_few_f1', test_few_f1, step)        

        loss_sum, loss_cnt = 0, 0
        last_log_time = time.time()
        if time.time() - last_save_time > config.checkpoint_interval:
            save_model(model, f'{config.task_name}_{start_time}', f'model_{step_no:05}.pt')
            save_model(opt, f'{config.task_name}_{start_time}', f'opt.pt')
            logger.info(f'step_no: {step_no}, save checkpoint!')
            last_save_time = time.time()


def train():
    global logger, batch_cnt
    logger = Logger(
        config.task_name,
        format_str='%(asctime)s - %(message)s',
        file_path=os.path.join(cs.LOG_DIR, f'{config.task_name}_{start_time}.log')
    )
    log_config(logger, vars(config))
    
    model = get_model()
    if config.last_training_time == 0 and cs.rank == 0:
        os.makedirs(os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{start_time}'), exist_ok=True)
        with open(os.path.join(cs.SAVED_MODEL_DIR, f'{config.task_name}_{start_time}', f'model_conf.pkl'), 'wb') as f:
            pickle.dump(model.config, f)
    model = to_device(model)

    opt = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': config.init_lr, 'initial_lr': config.init_lr}],
        lr=config.init_lr,
    )
    if config.last_training_time != 0:
        opt.load_state_dict(torch.load(
            os.path.join(
                cs.SAVED_MODEL_DIR,
                f'{config.task_name}_{config.last_training_time}',
                f'opt.pt'
            ), map_location=torch.device("cpu")
        ))

    scheduler = get_constant_schedule_with_warmup(opt, config.num_warmup_steps, config.last_step)

    training_set = get_training_set()
    dataloader, sampler = get_training_set_loader(
        training_set,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers
    )
    
    save_pkl_root = '/home/datamerge/ACL/Data/210422/pkl/'
    nor2len_train_part_dict = pickle.load(open(save_pkl_root+'nor2len_train_part_dict.pkl', 'rb'))
    afid2nor = pickle.load(open(save_pkl_root+"afid2nor.pkl", "rb"))
    id_to_cls = pickle.load(open('/home/jxqi/ACL/experiment/ann/dataset/id_to_cls.pkl', 'rb'))
    cls_to_id = { v:k for k,v in id_to_cls.items() }
    weights = [(1./nor2len_train_part_dict[afid2nor[cls_to_id[i]]])**(config.reweight_exp) for i in range(len(cls_to_id))]
    weights = torch.tensor(weights).to(torch.device("cuda"))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    try:
        for data in generate_data(dataloader, sampler):
            loss = train_one_batch(
                model=model,
                loss_fn=loss_fn,
                opt=opt,
                data=data,
            )
            after_batch(model, opt, loss, scheduler)
    except (StopTraining, KeyboardInterrupt):
        logger.info('Stop training.')
