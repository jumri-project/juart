# QRAGE

This repository provides a Pulseq implementation of the QRAGE MRI sequence, as described in our publication:  
["QRAGE—Simultaneous multiparametric quantitative MRI of water content, T1, T2*, and magnetic susceptibility at ultrahigh field strength"](https://onlinelibrary.wiley.com/doi/10.1002/mrm.30272).

# How to Use This Repository

Follow these steps to set up and use QRAGE:

1. **Clone the QRAGE Repository**

   ```bash
   git clone https://github.com/jumri-project/qrage.git
   cd qrage
   ```

2. **Install the Custom PyPulseq**

   QRAGE relies on modifications to PyPulseq that have not yet been merged upstream. You can install the custom PyPulseq in one of two ways:

   - **Option A: Clone and Install Locally**

     ```bash
     git clone https://github.com/jumri-project/pypulseq.git
     cd pypulseq
     git checkout develop
     pip install -e .
     ```

   - **Option B: Install Directly via pip**

     ```bash
     pip install git+https://github.com/jumri-project/pypulseq.git@devel
     ```

This repository uses Jupyter notebooks as examples.

**Note:** Git doesn’t handle diffing and merging notebooks very well by default. To improve this, you can configure git to use [nbdime](https://nbdime.readthedocs.io/en/latest/). Run the following command to enable nbdime globally:

> ```bash
> nbdime config-git --enable --global
> ```
> For more details, see the [nbdime git integration quickstart](https://nbdime.readthedocs.io/en/latest/#git-integration-quickstart).

## Differences from the Published Implementation

While we have worked diligently to replicate the QRAGE sequence originally developed in the SIEMENS IDEA framework, there are some minor differences between this implementation and the one used in the publication. For instance, the repetition time, inversion time, and echo times differ slightly. However, our comparisons indicate that these variations do not have a significant impact on image quality or the accuracy of the parameter maps.

## Missing Content

Currently, this repository contains only the sequence implementation. We are actively working on open-sourcing the complete image reconstruction pipeline—including both pre-processing and post-processing steps—which will be available soon.

## How to give credit

If you use this package, please acknowledge our work by citing:

Zimmermann M, Abbas Z, Sommer Y, et al. QRAGE—Simultaneous multiparametric quantitative MRI of water content, T1, T2*, and magnetic susceptibility at ultrahigh field strength. Magn Reson Med. 2025; 93(1): 228-244. doi: 10.1002/mrm.30272

A BibTeX file is directly contained within this package (QRAGE.bib).
