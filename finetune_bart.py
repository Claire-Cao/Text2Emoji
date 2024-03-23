import argparse
import os
from functools import partial

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from transformers import BartForConditionalGeneration

from utils import set_random_seed, shift_tokens_right
from dataset import Emoji2TextDataset


def train_loop(data_loader, model, loss_fn, optimizer, lr_scheduler, device):

    model.train()
    for batch_ind, batch_data in enumerate(data_loader):
        for k in batch_data:
            if isinstance(batch_data[k], torch.Tensor):
                batch_data[k] = batch_data[k].to(device)
            else:
                raise TypeError('data type is not supported')

        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(batch_data['target_ids'],
                                               data_loader.dataset.tokenizer.pad_token_id)

        # Run the model and get the logits
        outputs = model(batch_data['input_ids'],
                        attention_mask=batch_data['input_attention_mask'],
                        decoder_input_ids=decoder_input_ids,
                        use_cache=False)
        lm_logits = outputs[0]
        # Calculate the loss on the un-shifted tokens
        total_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]),
                             batch_data['target_ids'].view(-1))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print loss
        if device.index == 0 and batch_ind % 20 == 0:
            print("[current: {}/{}], loss: {}".format(batch_ind+1, len(data_loader), total_loss))


def test_loop(data_loader, model, loss_fn, device):
    
    model.eval()
    
    print('The total number of validation samples are {}'.format(len(data_loader)))
    total_loss = 0.0
    with torch.no_grad():
        for batch_ind, data in enumerate(data_loader):
            # print("processing {}th / {} image".format(batch_ind+1, len(data_loader)))
            input_ids = data['input_ids'].to(device)
            input_attention_mask = data['input_attention_mask'].to(device)
            target_ids = data['target_ids'].to(device)
            decoder_input_ids = shift_tokens_right(target_ids,
                                                   data_loader.dataset.tokenizer.pad_token_id)

            outputs = model(input_ids, input_attention_mask, decoder_input_ids=decoder_input_ids)
            lm_logits = outputs[0]
            batch_loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), target_ids.view(-1))
            print("processing {}th / {} image, loss: {}".format(batch_ind+1, len(data_loader), batch_loss))
            total_loss += batch_loss
    print('Validation Loss: {}'.format(total_loss / len(data_loader)))


def run_train(cfg, rank):
    # Distributed function called by all processes

    device = torch.device('cuda:{}'.format(rank))

    print("Building Data Loader ...")
    # training data loader
    train_dataset = Emoji2TextDataset(cfg['data_file_path'], cfg['tokenizer_file_dir'], is_training=True)
    sampler = DistributedSampler(train_dataset, cfg['world_size'], rank, shuffle=True)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg['samples_per_gpu'],
        sampler=sampler,
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        drop_last=True)

    print('Training data loader: there are {} batches/iterations in each epoch'.
          format(len(train_data_loader)))

    if rank == 0:
        # validation data loader
        val_dataset = Emoji2TextDataset(cfg['data_file_path'], cfg['tokenizer_file_dir'], is_training=False)
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4)
        print('Validation data loader: there are {} samples in each loader'.
              format(len(val_data_loader)))

    bart_model = BartForConditionalGeneration.from_pretrained(cfg['tokenizer_file_dir'])
    bart_model.resize_token_embeddings(len(train_dataset.tokenizer))
    loss_module = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id)
    bart_model = bart_model.to(device)
    nn.parallel.DistributedDataParallel(bart_model,
                                        device_ids=[rank],
                                        find_unused_parameters=True)

    # init optimizer
    optimizer = torch.optim.AdamW(bart_model.parameters(), lr=cfg['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                       max_lr=cfg['learning_rate'],
                                                       total_steps=cfg['num_epochs'] * len(train_data_loader),
                                                       pct_start=0.05)

    print("Start training loop...")
    for epoch in range(cfg['num_epochs']):
        print("Epoch {}\n------------------------".format(epoch+1))
        print("Running train loop ...")
        train_data_loader.sampler.set_epoch(epoch)  # Only for distributed training
        train_loop(train_data_loader,
                   bart_model,
                   loss_module,
                   optimizer,
                   lr_scheduler,
                   device)

        if rank == 0:
            if (epoch + 1) % 5 == 0:
                test_loop(val_data_loader,
                          bart_model,
                          loss_module,
                          device)
                print("Saving checkpoint ...")
                bart_model.save_pretrained("models/bart-base-emoji-epoch{}".format(epoch), from_pt=True)


def init_process(cfg, rank, fn, backend='nccl'):
    set_random_seed(cfg['random_seed'])
    world_size = cfg['world_size']
    torch.backends.cudnn.deterministic = cfg['deterministic']
    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(cfg, rank)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_path', type=str, default='./dataset/Text2Emoji/text2emoji.csv')
    parser.add_argument('--tokenizer_file_dir', type=str, default='./pretrain-models/bart-base-added-token')
    parser.add_argument('--random_seed', type=int, default=2024,
                        help='random seed to be used (also set in torch & numpy)')
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--workers_per_gpu', type=int, default=4)
    parser.add_argument('--samples_per_gpu', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--num_epochs', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_dict = vars(args)

    world_size = args.world_size
    processes = []
    mp.set_start_method("spawn")
    for rank_id in range(world_size):
        p = mp.Process(target=init_process, args=(cfg_dict, rank_id, run_train))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
