# Automated recognition by multiple convolutional neural networks of modern, fossil, intact and damaged pollen grains

This repository contains code, datasets, and instructions associated with the article:

Bourel B.<sup>1*</sup>, Marchant R.<sup>1</sup>, de Garidel-Thoron T.<sup>1</sup>, Tetard M.<sup>1</sup>, Barboni D.<sup>1</sup>, Gally Y.<sup>1</sup>, Beaufort L.<sup>1</sup> (2020). Automated recognition by multiple convolutional neural networks of modern, fossil, intact and damaged pollen grains. *Computers & Geosciences*, *140*, 104498.

1: CEREGE, Aix Marseille Université, CNRS, IRD, INRA, Coll. France, Technopole Arbois, 13545 Aix en Provence cedex 4, France

\* Corresponding author: benjamin.bourel@inria.fr

**DOI of paper**: <https://doi.org/10.1016/j.cageo.2020.104498>

## Overview

This project provides scripts to classify pollen grains using multiple Convolutional Neural Networks (CNNs). The models are trained and evaluated on datasets of **intact**, **damaged**, and **fossil** pollen grains. Single- and multi-CNN architectures are included, with and without data augmentation.

## Repository Structure

```         
├── CNN_test_v2.1.0.py                     # Single CNN with augmentation
├── CNN_test_v2.1.0(without_augm).py       # Single CNN without augmentation
├── Multi-CNN_test_V3.5.2.py               # Multi-CNN with augmentation
├── Multi-CNN_test_V3.5.2(without_augm).py # Multi-CNN without augmentation
├── CNN_solo/                              # Single CNN model parameters & training logs
├── intact/                                # Intact pollen dataset
├── damaged/                               # Damaged pollen dataset
├── fossil/                                # Fossil pollen dataset
└── User_manual.pdf                        # Detailed usage instructions
```
