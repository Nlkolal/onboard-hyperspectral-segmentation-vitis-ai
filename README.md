# Vitis AI Hyperspectral Segmentation

## Overview
This repository contains the implementation and FPGA deployment of deep learning models for onboard land–sea–cloud segmentation of hyperspectral satellite imagery. The work focuses on deploying convolutional neural network models using the Vitis AI framework and its Deep Processing Unit (DPU) for efficient inference on resource-constrained platforms.

The project is motivated by the need for intelligent onboard data reduction in CubeSat missions, where hyperspectral sensors generate large data volumes that exceed downlink capacity.

## Models
The following architectures are implemented and evaluated:
- C-UNet
- C-UNet++
- C-FCN
- UNet-Small
- Justo-UNet-Simple

Models are trained on hyperspectral images acquired by the HYPSO-1 CubeSat developed at NTNU.

## Hardware and Deployment
- **Target platform:** Xilinx Zynq UltraScale+ MPSoC ZCU104  
- **Acceleration framework:** Vitis AI  
- **Inference engine:** Deep Processing Unit (DPU)  
- **Quantization:** Post-training quantization (INT8)

## Dataset
Training and evaluation are performed using a labeled dataset consisting of 44 hyperspectral scenes from the HYPSO-1 mission. Dimensionality reduction is applied using principal component analysis (PCA) prior to inference.

*Dataset access details or links may be provided if permitted.*

## Results
- Four principal components capture over **99.5%** of the spectral variance  
- Post-training quantization introduces minimal accuracy degradation  
- **Justo-UNet-Simple** achieves the best trade-off between accuracy, throughput, and resource usage  
- Real-time inference is achieved on FPGA hardware

## Thesis Reference
This repository accompanies the master’s project:

**Onboard Cloud–Land–Sea Segmentation Using Vitis AI on the HYPSO CubeSat**  
Norwegian University of Science and Technology (NTNU), 2025
