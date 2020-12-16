from model import AtnCRNN
from data import CommandsDataset, my_collate, my_sampler, Noiser, LogMelSpectrogram

from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import random
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_ as clip




def set_seed(n):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed = n
    random.seed(n)
    np.random.seed(n)


def FAFR_scores(preds, targets):
    FA = sum(preds[targets == 0])/len(targets)
    FR = sum(targets[preds == 0])/len(targets)
    return FA, FR


if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datadir = "speech_commands"

    samples_by_target = {
        cls: [os.path.join(datadir, cls, name) for name in os.listdir("./speech_commands/{}".format(cls))]
        for cls in os.listdir(datadir)
        if os.path.isdir(os.path.join(datadir, cls))
    }
    print('Classes:', ', '.join(sorted(samples_by_target.keys())[1:]))

    names = []
    words = []
    lbls = []
    noises = []
    for wv in tqdm(samples_by_target.keys()):
        if wv != '_background_noise_':
            for sample in samples_by_target[wv]:
                word = sample.split('/')[1]
                if word == 'sheila':
                    lbl = 1
                else:
                    lbl = 0
                names.append(sample)
                words.append(word)
                lbls.append(lbl)
        else:
            for noise in samples_by_target[wv]:
                if 'README' not in noise:
                    noises.append(noise)
    names = np.array(names)
    lbls = np.array(lbls)
    words = np.array(words)

    root_dir = '/content'
    train_len = int(len(lbls) * 0.8)

    idxs = np.array(range(len(lbls)))
    np.random.shuffle(idxs)

    train_names = names[idxs[:train_len]]
    train_lbls = lbls[idxs[:train_len]]
    train_words = words[idxs[:train_len]]
    train = CommandsDataset(root_dir, train_names, train_lbls, train_words, transform=Noiser(noises_list=noises))
    # train, val = torch.utils.data.random_split(dataset, (train_len, val_len))

    val_names = names[idxs[train_len:]]
    val_lbls = lbls[idxs[train_len:]]
    val_words = words[idxs[train_len:]]
    val = CommandsDataset(root_dir, val_names, val_lbls, val_words, transform=Noiser(noises_list=noises))

    train_sampler = my_sampler(train_lbls)
    val_sampler = my_sampler(val_lbls)

    batch_size = 256
    n_mels = 40

    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler, collate_fn=my_collate)

    val_loader = DataLoader(val, batch_size=batch_size, sampler=val_sampler, collate_fn=my_collate)

    melspec = LogMelSpectrogram(n_mels=n_mels).to(device)
    melspec_val = LogMelSpectrogram(n_mels=n_mels, masking=False).to(device)

    model = AtnCRNN(n_mels, 128, 2, device).to(device)
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epochs = 40
    pbar = tqdm_notebook(total=epochs * (len(train) // batch_size))
    for epoch in range(epochs):
        running_loss = 0.0
        val_loss = 0.0
        ctr = 0
        val_ctr = 0
        model.train()
        for batch in train_loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            X_mel = melspec(X)

            optimizer.zero_grad()

            pred = model(X_mel)
            loss = F.nll_loss(pred, y)
            running_loss += loss.item()
            loss.backward()
            clip(model.parameters(), 5)

            optimizer.step()
            ctr += 1
            pbar.update(1)

        model.eval()
        val_preds = []
        true = []
        for batch in val_loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            X_mel = melspec_val(X)
            with torch.no_grad():
                pred = model(X_mel)
                loss = F.nll_loss(pred, y)
                val_loss += loss.item()
            pred = (torch.argmax(pred, dim=1)).cpu()
            val_preds.extend(pred.tolist())
            true.extend(y.cpu().tolist())
            val_ctr += 1
        val_preds = np.array(val_preds)
        true = np.array(true)
        FA, FR = FAFR_scores(val_preds, true)
        acc = accuracy_score(true, val_preds)
        print('train loss', running_loss / ctr, 'val loss', val_loss / val_ctr, 'val FA', FA, 'val FR', FR, 'val acc',
              acc)

    pbar.close()
