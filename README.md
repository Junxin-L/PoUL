# README.md for PoUL

## Overview

PoUL (Proof of Unlearning) is a novel framework designed for Machine-Learning-as-a-Service (MLaaS) providers, addressing the critical need for machine unlearning. This concept has gained prominence with the increasing concerns about data privacy in MLaaS contexts. PoUL uniquely combines backdoor triggers and incremental learning to provide a verifiable machine unlearning scheme that does not compromise on performance and service quality.

## Key Features

- **Backdoor-Assisted Validation**: Utilizes backdoor triggers to embed invisible markers in privacy-sensitive data, preventing MLaaS providers from distinguishing poisoned data and spoofing validation.
- **Incremental Learning Integration**: Incorporates an efficient incremental learning approach with an index structure, facilitating performance in retraining post-data deletion.
- **Data Deletion Compliance Verification**: Enables users to verify if providers have complied with data deletion requests using prediction results.
- **Retaining Service Quality**: Ensures that the machine unlearning process does not degrade the overall service quality or performance of MLaaS.

## Requirements

- Python 3.x
- Dependencies as specified in `requirements.txt` (if provided)

## Installation

Download and extract the PoUL package to your desired directory.

## Usage

The PoUL framework consists of several Python modules:

- `Model.py`: Defines the ML model structure.
- `allocate.py`: Manages client data allocation.
- `backdoor.py`: Implements the backdoor trigger mechanism.
- `backdoor_utils.py`: Provides utility functions for the backdoor process.
- `class_name_convert.py`: Converts class names for compatibility.
- `data_utils.py`: Contains data processing utilities.
- `predict_utils.py`: Utilities for making predictions.
- `size_convert.py`: Handles size conversion.
- `train.py`: Main training module.
- `train_utils.py`: Provides training-related utilities.

To start using PoUL, run the `train.py` script:

```bash
python train.py
