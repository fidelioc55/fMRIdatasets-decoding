# Linear Contrastive Alignment for Brain Decoding

This repository contains the code accompanying the paper  
**"Linear Maps, Contrastive Objectives: A Principled Strategy for fMRI Decoding Across Modalities"**.  
The project investigates how functional MRI (fMRI) activity can be aligned with the embedding spaces of foundation models in **vision**, **language**, and **audio** using **linear contrastive objectives**.

---

## Repository Structure

### 📓 Notebooks
- **`NSD_img.ipynb`**  
  End-to-end pipeline for the **NSD (Natural Scenes Dataset)** image experiments.  
  Handles loading preprocessed fMRI data, CLIP image embeddings, model training, and evaluation with retrieval metrics.

- **`GTZAN_music.ipynb`**  
  End-to-end pipeline for the **GTZAN music dataset** experiments.  
  Loads fMRI and CLAP audio embeddings, performs preprocessing, runs contrastive training, and evaluates retrieval performance.

- **`HUTH_lang.ipynb`**  
  End-to-end pipeline for the **Lebel et al. (language dataset)** experiments.  
  - Parses aligned speech–text data using TextGrids.  
  - Builds word-, and sentence-level feature sequences.  
  - Encodes sentences with LLAMA embeddings.  
  - Trains linear contrastive decoders to align brain activity with semantic embeddings.  

---

### 📜 Scripts

#### Image & Music
- **`wandb_img.py`**  
  Scripted version of the NSD image experiments with **Weights & Biases (wandb)** sweeps.  
  Implements grid search over hyperparameters and logs training/evaluation metrics.

- **`wandb_music.py`**  
  Scripted version of the GTZAN music experiments with **wandb** sweeps.  
  Includes fMRI preprocessing, alignment, training, and retrieval evaluation.

- **`controt_network.py`**  
  Core model definition: **Contrastive-Model** (PyTorch Lightning).  
  Provides the encoder, contrastive loss, and training loop used across modalities.

#### Language (HUTH dataset)
- **`HUTH_textgrid.py`**  
  Parser for **Praat TextGrid** annotation files. Extracts word, phoneme, and interval-level transcripts.

- **`HUTH_stimulus_utils.py`**  
  Utilities for handling stimulus presentation logs and TextGrids.  
  - Loads time-locked alignments (TR files).  
  - Simulates TRs.  
  - Aligns sound onset/offset times with fMRI triggers.

- **`HUTH_npp.py`**  
  Simple NumPy preprocessing helpers (demeaning, z-scoring, rescaling, correlations).

- **`HUTH_interpdata.py`**  
  Time-series interpolation utilities (linear, sinc, Lanczos, Gabor, exponential).  
  Used to resample and align stimulus features with fMRI TRs.

- **`HUTH_dsutils.py`**  
  Converters for building **DataSequence** objects from transcripts.  
  Supports word, phoneme, character, and dialogue sequences, plus phoneme histograms and semantic feature models.

- **`HUTH_SemanticModel.py`**  
  Sentence-level semantic encoder based on **CLAP**.  
  - Encodes sentences or rolling context windows.  
  - Returns embeddings aligned with word/phrase timings.  
  - Output can be wrapped into a `DataSequence` for alignment with fMRI.

- **`DataSequence.py`**  
  Core utility for handling sequential data aligned to TRs.  
  - Stores data, split indices, and time alignments.  
  - Provides chunking, averaging, downsampling, and construction from TextGrids.  
  - Serves as the main interface between text annotations and brain recordings.

---

