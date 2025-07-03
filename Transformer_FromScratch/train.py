from model import build_transformer
from dataset import TranslationDataset, build_causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# decode an output sequence using greedy decoding
# builds the output one token at a time until EOS or max_len is hit
def simple_greedy_decode(model, src_seq, src_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    enc_output = model.encode(src_seq, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src_seq).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # generate mask and get decoder output
        tgt_mask = build_causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        dec_out = model.decode(enc_output, src_mask, decoder_input, tgt_mask)

        # predict next token
        logits = model.project(dec_out[:, -1])
        _, next_token = torch.max(logits, dim=1)

        # append to sequence
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).fill_(next_token.item()).type_as(src_seq).to(device)
        ], dim=1)

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)


# run evaluation on the validation set and log predictions/metrics
def evaluate(model, val_loader, src_tokenizer, tgt_tokenizer, max_len, device, log_fn, step, writer, sample_count=2):
    model.eval()
    count = 0
    original = []
    ground_truth = []
    generated = []

    # get terminal width if possible
    try:
        _, width = os.popen('stty size', 'r').read().split()
        width = int(width)
    except:
        width = 80

    with torch.no_grad():
        for batch in val_loader:
            count += 1
            src_input = batch["encoder_input"].to(device)
            src_mask = batch["encoder_mask"].to(device)

            assert src_input.size(0) == 1, "Validation batch size must be 1"

            # decode prediction
            pred_ids = simple_greedy_decode(model, src_input, src_mask, src_tokenizer, tgt_tokenizer, max_len, device)

            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            pred_text = tgt_tokenizer.decode(pred_ids.detach().cpu().numpy())

            # save results
            original.append(src_text)
            ground_truth.append(tgt_text)
            generated.append(pred_text)

            log_fn('-' * width)
            log_fn(f"{'SOURCE:':>12}{src_text}")
            log_fn(f"{'TARGET:':>12}{tgt_text}")
            log_fn(f"{'PREDICTED:':>12}{pred_text}")

            if count == sample_count:
                log_fn('-' * width)
                break

    if writer:
        # log validation metrics
        cer = torchmetrics.CharErrorRate()(generated, ground_truth)
        wer = torchmetrics.WordErrorRate()(generated, ground_truth)
        bleu = torchmetrics.BLEUScore()(generated, ground_truth)

        writer.add_scalar('validation cer', cer, step)
        writer.add_scalar('validation wer', wer, step)
        writer.add_scalar('validation BLEU', bleu, step)
        writer.flush()


# yield all sentences in a specific language from dataset
def stream_sentences(data, lang_key):
    for row in data:
        yield row['translation'][lang_key]


# initialize or load a tokenizer for a given language
def init_or_load_tokenizer(cfg, dataset, language):
    tok_path = Path(cfg['tokenizer_file'].format(language))
    if not tok_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(stream_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tok_path))
    else:
        tokenizer = Tokenizer.from_file(str(tok_path))
    return tokenizer


# load dataset, split it, prepare dataloaders and tokenizers
def load_data(cfg):
    raw_data = load_dataset(cfg['datasource'], f"{cfg['lang_src']}-{cfg['lang_tgt']}", split='train[:2000]')

    tokenizer_src = init_or_load_tokenizer(cfg, raw_data, cfg['lang_src'])
    tokenizer_tgt = init_or_load_tokenizer(cfg, raw_data, cfg['lang_tgt'])

    # split into train and validation sets
    train_len = int(0.9 * len(raw_data))
    val_len = len(raw_data) - train_len
    train_raw, val_raw = random_split(raw_data, [train_len, val_len])

    # wrap with TranslationDataset class
    train_set = TranslationDataset(train_raw, tokenizer_src, tokenizer_tgt, cfg['lang_src'], cfg['lang_tgt'], cfg['seq_len'])
    val_set = TranslationDataset(val_raw, tokenizer_src, tokenizer_tgt, cfg['lang_src'], cfg['lang_tgt'], cfg['seq_len'])

    # print maximum sentence lengths
    max_src_len, max_tgt_len = 0, 0
    for row in raw_data:
        max_src_len = max(max_src_len, len(tokenizer_src.encode(row['translation'][cfg['lang_src']]).ids))
        max_tgt_len = max(max_tgt_len, len(tokenizer_tgt.encode(row['translation'][cfg['lang_tgt']]).ids))

    print(f'Max source length: {max_src_len}')
    print(f'Max target length: {max_tgt_len}')

    return (DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True),
            DataLoader(val_set, batch_size=1, shuffle=True),
            tokenizer_src, tokenizer_tgt)


# build the transformer model with specified vocab sizes
def build_model(cfg, src_vocab_size, tgt_vocab_size):
    return build_transformer(src_vocab_size, tgt_vocab_size, cfg['seq_len'], cfg['seq_len'], d_model=cfg['d_model'])


# main training loop
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    # ensure output folder exists
    Path(f"{cfg['datasource']}_{cfg['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tok_src, tok_tgt = load_data(cfg)
    model = build_model(cfg, tok_src.get_vocab_size(), tok_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(cfg['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-9)

    epoch_start = 0
    global_step = 0

    # check for pretrained weights
    ckpt_file = latest_weights_file_path(cfg) if cfg['preload'] == 'latest' else get_weights_file_path(cfg, cfg['preload']) if cfg['preload'] else None

    if ckpt_file:
        print(f'Loading checkpoint: {ckpt_file}')
        state = torch.load(ckpt_file)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch_start = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print('Starting from scratch...')

    loss_func = nn.CrossEntropyLoss(ignore_index=tok_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(epoch_start, cfg['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in progress:
            src = batch['encoder_input'].to(device)
            tgt = batch['decoder_input'].to(device)
            src_mask = batch['encoder_mask'].to(device)
            tgt_mask = batch['decoder_mask'].to(device)

            # run transformer forward pass
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)
            logits = model.project(dec_out)
            targets = batch['label'].to(device)

            # compute loss
            loss = loss_func(logits.view(-1, tok_tgt.get_vocab_size()), targets.view(-1))
            progress.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # evaluate model after each epoch
        evaluate(model, val_loader, tok_src, tok_tgt, cfg['seq_len'], device, lambda m: progress.write(m), global_step, writer)

        # save checkpoint
        ckpt_path = get_weights_file_path(cfg, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, ckpt_path)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train(config)
