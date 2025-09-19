import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, act_fn=nn.Identity, alignment_layers_keys=[1,2,5,7], common_dim=1024):
        super(Encoder, self).__init__()
        self.common_dim=common_dim
        self.alignment_layers={}
        for k in alignment_layers_keys:
            self.alignment_layers[k]=nn.Linear(input_dim, common_dim)  
        
        layers = []
        prev_dim = input_dim
        layers.append(nn.LayerNorm(common_dim))
        if len(hidden_dims):

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(common_dim, hidden_dim))
                layers.append(act_fn())     
                # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
           
        else:
            layers.append(act_fn()) 
            layers.append(nn.Linear(common_dim, output_dim))
        self.net = nn.Sequential(*layers)   

        
    def _apply(self, fn):
        super(Encoder, self)._apply(fn)        
        for k,v in self.alignment_layers.items():
            self.alignment_layers[k]._apply(fn)
            
    
    def forward(self, x, k=None):
        
        def apply_alignment_layers(x, k, alignment_layers):
            # Create an empty tensor to store the results
            result = torch.empty_like(x)
            result = result[:,:self.common_dim]
            
            # Iterate through each unique key in k
            for key in k.unique():
                # Create a mask for all elements that match the current key
                mask = (k == key.item())
                
                # Apply the corresponding alignment layer to the masked elements
                result[mask] = alignment_layers[key.item()](x[mask])
            
            return result
        
        if k is None:
            k=torch.ones(len(x))
        # Apply alignment layers to x using the custom function
        x = apply_alignment_layers(x, k, self.alignment_layers)
        # x = self.alignment_layers[k](x)
        return self.net(x)


# Funzione per calcolare la similarità coseno
def cosine_similarity_matrix(A, B):
    A_norm = F.normalize(A, dim=1)
    B_norm = F.normalize(B, dim=1)
    return torch.mm(A_norm, B_norm.T)

# NT-Xent Loss (Contrastive)
def contrastive_loss_nt(S, tau):
    S_exp = torch.exp(S / tau)
    loss = -torch.log(torch.diag(S_exp) / S_exp.sum(dim=1))
    return loss.mean()

def contrastive_loss(z_i, z_j, tau):
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    logits = (z_i @ z_j.T) / tau
    similarities = z_j @ z_j.T
    # targets = torch.nn.functional.softmax(similarities * self.temperature, dim=-1)
    targets = torch.arange(logits.shape[0]).long().to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    return loss


class ContrastiveOTModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, output_dim, common_dim=1024, alignment_layers_keys=[1,2,5,7], act_fn=nn.Identity, tau=0.2, lr=1e-3, wd=1e-4, loss_weights=(1, 1)):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, output_dim, act_fn=act_fn, alignment_layers_keys=alignment_layers_keys, common_dim=common_dim) 
        self.tau = tau
        self.lr = lr
        self.wd = wd
        self.loss_weights = loss_weights

        self.train_losses = []
        self.train_ot_loss=[]
        self.train_contrastive_loss=[]
        self.val_losses = []
        self.val_ot_loss=[]
        self.val_contrastive_loss=[]
    
        self.train_history={}
        self.val_history={}

        self.train_history["train_loss"]=[]
        self.train_history["train_ot_loss"]=[]
        self.train_history["train_contrastive_loss"]=[]
        self.val_history["val_loss"]=[]
        self.val_history["val_ot_loss"]=[]
        self.val_history["val_contrastive_loss"]=[]

    def forward(self, x, embeddings = None, **kwargs):
        fmri_proj = self.encoder(x, **kwargs)
        if embeddings is not None:
            embeddings_proj = embeddings  
            return fmri_proj, embeddings_proj
        return fmri_proj

    def loss_fn(self, x_latent, y_latent):
        S = cosine_similarity_matrix(x_latent, y_latent)
        contrastive_loss_value = contrastive_loss_nt(S, self.tau) 
        total_loss = self.loss_weights[1]*contrastive_loss_value
        return total_loss, contrastive_loss_value
    
    def compute_loss(self, x, y, idx):
        x_latent = self(x, k=idx)
        y_latent = y  
        return self.loss_fn(x_latent, y_latent)

    def training_step(self, batch, batch_idx):
        x, y, idx = batch 
        loss, contrastive_loss_value = self.compute_loss(x, y, idx)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_contrastive_loss", contrastive_loss_value, on_epoch=True)

        self.train_losses.append(loss.item())
        self.train_contrastive_loss.append(contrastive_loss_value.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch 
        loss, contrastive_loss_value = self.compute_loss(x, y, idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_contrastive_loss", contrastive_loss_value, on_epoch=True)

        self.val_losses.append(loss.item())
        self.val_contrastive_loss.append(contrastive_loss_value.item())
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.train_history["train_loss"].append(np.mean(self.train_losses))
        self.train_history["train_contrastive_loss"].append(np.mean(self.train_contrastive_loss))
        self.train_losses = []
        self.train_contrastive_loss=[]
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        self.val_history["val_loss"].append(np.mean(self.val_losses))
        self.val_history["val_contrastive_loss"].append(np.mean(self.val_contrastive_loss))
        self.val_losses = []
        self.val_contrastive_loss=[]
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, threshold=0.05, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    
