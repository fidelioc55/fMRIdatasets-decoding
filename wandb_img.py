import numpy as np
import nibabel as nib
import nilearn 
import matplotlib.pyplot as plt
import os
from os.path import join as opj
import pandas as pd
import seaborn as sns
import glob
from nilearn import plotting
from nilearn.image import *
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
from nilearn.plotting import plot_img, plot_epi
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from controt_network import ContrastiveOTModel
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torchvision import transforms
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity



dest_path = "/srv/nfs-data/sisko/ot_datasets/NSD"
os.makedirs(dest_path,exist_ok=True)
device = "cuda:2"

train_data = np.load(f"{dest_path}/train_data.npy")
test_data = np.load(f"{dest_path}/test_data.npy")
train_clip_img_embeds = torch.load(f"{dest_path}/train_clip_img_embeds.pt")
test_clip_img_embeds = torch.load(f"{dest_path}/test_clip_img_embeds.pt")
# img_train = torch.load(f"{dest_path}/img_train.pt")
# img_test = torch.load(f"{dest_path}/img_test.pt")
# train_captions = torch.load(f"{dest_path}/train_captions.pt")
# test_captions = torch.load(f"{dest_path}/test_captions.pt")
subject_train_ids = np.load(f"{dest_path}/subject_train_ids.npy")
subject_test_ids = np.load(f"{dest_path}/subject_test_ids.npy")
subject_train_ids=[int(i[-1]) for i in subject_train_ids]
subject_test_ids=[int(i[-1]) for i in subject_test_ids]

train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data).float(),train_clip_img_embeds.squeeze(1).float(), torch.tensor(subject_train_ids))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data).float(),test_clip_img_embeds.squeeze(1).float(), torch.tensor(subject_test_ids))

sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "act_fn": {"values": ["nn.Identity", "nn.ReLU", "nn.GELU"]},
        "common_dim": {"values": [15724]},
        "hidden_dims": {"values": [[1024], []]},
        "latent_dim": {"values": [512]},
        "lr": {"values": [1e-4]},
        "tau": {"values": [0.02, 0.10]},
        "wd": {"values": [1e-3]},
        "batch_size": {"values": [2048, 512]},
        "loss_weights": {"values": [[1, 1], [0, 1], [0.7, 0.3], [0.3, 0.7]]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="NSD_otcl_final")

def train():
    wandb.init()
    config = wandb.config

    BS = config.batch_size 
    BS_test = int(BS / 2) 
    clip_train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    clip_test_dataloader = DataLoader(test_dataset, batch_size=BS_test, shuffle=False)

    brain_model = ContrastiveOTModel(
        input_dim=15724,
        hidden_dims=config.hidden_dims,
        output_dim=config.latent_dim,
        common_dim=config.common_dim, 
        alignment_layers_keys=[1, 2, 5, 7],
        act_fn=eval(config.act_fn),
        tau=config.tau,
        lr=config.lr,
        wd=config.wd,
        loss_weights=config.loss_weights,
    )

    pl.seed_everything(55, workers=True)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.09, patience=9, verbose=True, mode="min")
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=100, devices=[2], callbacks=[early_stop_callback], logger=wandb_logger)
    trainer.fit(brain_model, clip_train_dataloader, clip_test_dataloader)

    wandb.log({
        "val_loss": trainer.callback_metrics["val_loss"],
        "train_loss": brain_model.train_history["train_loss"][-1],
        "train_contrastive_loss": brain_model.train_history["train_contrastive_loss"][-1],
        "val_contrastive_loss": brain_model.val_history["val_contrastive_loss"][-1]
    })

    results = defaultdict(lambda: {"y_pred": [], "y_gt": [], "gt_images": []})
    with torch.no_grad():
        for batch in tqdm.tqdm(clip_test_dataloader):
            x, y, k = batch
            y_hat_embeddings = brain_model(x, k=k)
            for subject_index in torch.unique(k):
                mask = k == subject_index
                results[subject_index.item()]["y_pred"].append(y_hat_embeddings[mask])
                results[subject_index.item()]["y_gt"].append(y[mask])

    for subject_index in results:
        results[subject_index]["y_pred"] = torch.cat(results[subject_index]["y_pred"], dim=0)
        results[subject_index]["y_gt"] = torch.cat(results[subject_index]["y_gt"], dim=0)

    for subject_index, subject_data in results.items():
        y_pred_np = subject_data["y_pred"].cpu().numpy()
        y_gt_np = subject_data["y_gt"].cpu().numpy()
        similarity_matrix = cosine_similarity(y_pred_np, y_gt_np)
        top1_acc = (np.argmax(similarity_matrix, axis=1) == np.arange(len(y_gt_np))).mean()
        top5_acc = np.mean([1 if i in np.argsort(-similarity_matrix[i])[:5] else 0 for i in range(len(y_gt_np))])
        wandb.log({"Top-1 Accuracy": top1_acc, "Top-5 Accuracy": top5_acc})

    wandb.finish()

wandb.agent(sweep_id, function=train)
