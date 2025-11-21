# Speech Enhancement with CNNs

## Overview
This project aims to build a speech enhancement model using CNNs to remove noise from speech audio.

## Data
clean speech audio files from BBC Learning English Podcast, set mono, sample rate at 16000, wav format using Adobe Audition.
ambient noise files downloaded from freesound.com. 


## Key Steps
- Loading clean speech audio and noise audio. Mixing clean audio with noise on different SNR levels. Ensuring that the generated noisy speech is of the same length as the clean speech and corresponds to it in the lists.
- Performing STFT on all the clean audio and noisy audio and getting magnitude spectrograms. Segmenting the magnitude spectrograms into fixed-size, overlapping time patches. Each patch represents a short-time time–frequency context that can be processed by the CNN. Stacking the patches to create a new batch dimension. Transforming the patches into Tensors.

## Results  