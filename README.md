The simulation data used in the paper are archived at the Duke Research Data Repository: https://research.repository.duke.edu/record/515?&ln=en

# Leveraging Correlated Decoding for Bias-Tailored Clifford Deformed Compass Codes

This repository contains the numerical simulation framework and data analysis tools for "Leveraging Correlated Decoding for Bias-Tailored Clifford Deformed Compass Codes." This work explores the performance of Clifford Deformed (CD) compass codes under biased noise models with correlated decoding.

## 📌 Overview

Clifford Deformed (CD) Compass Codes provide a flexible architecture for adapting quantum error correction to asymmetric (biased) noise. This repository provides:
* **Code Generation:** Scripts to construct CD-deformed compass code stabilizers.
* **Correlated Decoding:** Implementation of correlated decoding with tailoring for biased noise.
* **Threshold Analysis:** Tools to calculate logical error rates and extract code thresholds using our (and Pymatching's) decoders.


### Prerequisites
* Python 3.8+
* `stim` (for circuit simulations)
* `pymatching` (or your preferred MWPM decoder)
* `numpy`, `scipy`, `matplotlib`

### Installation
```bash
git clone [https://github.com/acatspirit/correlated_decoding_cd_compass_codes.git](https://github.com/acatspirit/correlated_decoding_cd_compass_codes.git)
cd correlated_decoding_cd_compass_codes
pip install -r requirements.txt 
```

### Citation 
```bash
@article{
    author = {Meinking, Arianna and Campos, Julie and Brown, Kenneth R.}
    title = {LEveraging Correlated Decoding for Bias-Tailor Clifford Deformed Compass Codes}
    year = {2026}
    journal = {arXiv preprint. arXiv:(fill in)}
}
```