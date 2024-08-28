# Link Inference Attack in Vertical Federated Graph Learning

This repository contains the code and resources to reproduce the main results of the paper "Link Inference Attack in Vertical Federated Graph Learning".

## Requirements and Setup

To set up the environment and install all required libraries:

1. Run the `setup_lia_vfgl.sh` script:
   ```
   ./setup_lia_vfgl.sh
   ```
   This script will install the necessary environment with all required libraries and download all datasets used in the paper.

2. The main requirements are listed in `requirements.txt`, including:
   - PyTorch 1.13.0
   - PyTorch Geometric 2.2.0
   - NumPy 1.23.5
   - Pandas 1.5.2
   - Matplotlib 3.6.2
   - Wandb 0.13.6
   - Scikit-learn 1.2.0

3. Activate the environment before running any experiments:
   ```
   source lia_vfgl_env/bin/activate
   ```
   The name of the environment (lia_vfgl_env) should be apparent in your command prompt after activation.

## Hardware Requirements

Our results were primarily produced using the following setup:
- 4x NVIDIA GeForce RTX 3090 GPUs (24GB VRAM each)
- AMD EPYC 7272 12-Core Processor
- 125GB RAM
- Ubuntu 20.04.6 LTS with Linux kernel 5.4.0-177-generic

We have also tested our code for the Cora dataset on a smaller setup:
- "Compute VM" of Artifacts submission platform
- 16 core CPU
- 64GB memory
- Ubuntu 22.04

Note: The Cora dataset experiments do not require a GPU and can be run in a reasonable time on the smaller setup.

## Reproducing Results

To reproduce the results of the paper, follow these steps:

1. Navigate to the `scripts` directory.
2. Run the appropriate script for each experiment. Each script is associated with a specific experiment and will log results in the `log` directory.
3. After the script execution is complete, use one of the following methods to parse logs and produce final tables and graphs:
   a. Use the corresponding Jupyter notebook in the `reproduced-results` directory.
   b. Run the associated Python script in the same directory.

### Recommended Workflow

1. Go to the `reproduced-results` directory.
2. Read the `.ipynb` file associated with the experiment you want to reproduce.
3. Follow the detailed instructions in the notebook on how to run the corresponding script.
4. After the script has finished executing, choose one of these options:
   - Run the cells in the Jupyter notebook to generate the final results.
   - Execute the corresponding `.py` script to parse logs and save tables and figures.

## Experiments

The main results of the paper are presented in various tables and figures. Each experiment is associated with:

1. A script in the `scripts` directory that runs the experiment.
2. A log file in the `log` directory, named according to the experiment set in the script.
3. A Jupyter notebook in the `reproduced-results` directory for parsing logs and producing tables and graphs.

### Table 4 and Other Results

To reproduce Table 4 and other results from the paper:

1. Locate the corresponding script in the `scripts` directory.
2. Run the script following the instructions in the associated Jupyter notebook.
3. Once the script has finished, open the Jupyter notebook and execute its cells to generate the final results.

## Directory Structure

- `scripts/`: Contains the scripts for running each experiment.
- `log/`: Directory where experiment logs are stored.
- `reproduced-results/`: Contains Jupyter notebooks and Python scripts for parsing logs and producing final results.

## Execution Time Estimation

The execution time for each experiment varies based on the dataset. These estimates are for 5 parallel runs on our primary hardware setup. Note that each experiment requires multiple runs based on the number of data points needed and the number of seeds. You can reduce the total execution time by lowering the number of seeds.

Estimated execution times per 5 parallel runs:
- Cora and Citeseer: 3 minutes
- Amazon Photo: 22 minutes
- Amazon Computer: 77 minutes
- Twitch FR: 30 minutes
- Twitch DE: 55 minutes
- Twitch EN: 30 minutes

## Detailed Instructions

For detailed instructions on running each experiment, please refer to the corresponding Jupyter notebook in the `reproduced-results` directory.

If you prefer not to work with notebooks, each `.ipynb` file in the `reproduced-results` directory has a corresponding `.py` script. These scripts parse the logs and save the tables and figures directly in the directory, providing an alternative to using the notebooks.
