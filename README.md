# PINN-Sed

**PINN-Sed** is an interconnected Physics-Informed Neural Network (PINN) framework for simulating suspended sediment transport in river networks at high temporal resolution.

## Overview

This model embeds the governing sediment transport and advection–dispersion equations directly into neural networks. Multiple PINNs are coupled across river reaches through boundary conditions, allowing consistent prediction of sediment concentrations across a connected river system.

The framework reduces reliance on large datasets and heavy calibration while maintaining strong predictive performance.

## Features

* Physics-informed neural networks (PINNs) for sediment transport
* Coupled multi-reach river network modeling
* Incorporation of hydraulic variables (flow, depth, slope, stream power)
* PDE-constrained training with boundary and initial conditions
* GPU support via PyTorch

## Input Data

* Time series of suspended sediment concentration (Excel)
* Hydraulic variables (h, q, Q, slope, stream power)
* Configuration file (`config.toml`) for paths, parameters, and constants

## Model Structure

* Three neural networks representing interconnected river reaches
* Inputs: space (x), time (t), and discharge (q)
* Outputs: sediment concentration (c)
* Loss function combines:

  * PDE residual loss
  * Initial condition loss
  * Boundary condition loss
  * Coupling loss between reaches

## Output

* Trained model weights (`.pth` files)
* Loss history saved as CSV
* Predicted sediment concentrations

## How to Run

1. Update `config.toml` with correct paths and parameters
2. Prepare input Excel file with required columns
3. Run the main Python script
4. Monitor training loss in the console
5. Outputs are saved to the specified directories

## Applications

* River network sediment transport modeling
* Flood event sediment prediction
* Data-scarce environments
* Transferable to other catchments with similar data

## Reference

**PINN-Sed: An Interconnected Physics-Informed Machine Learning Model for High Resolution Suspended Sediment Transport in River Network** [Under review]


