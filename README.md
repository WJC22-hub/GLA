GLA: ...

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/你的文章链接)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

> **Abstract:** This repository contains the official implementation of the paper **"[Insert Your Paper Title Here]"**. We propose **GLA**, a novel backdoor attack method that achieves high Attack Success Rate (ASR) with low poisoning rates, maintaining stealthiness against state-of-the-art defenses.

<div align="center">
<img src="Fig/overview.png" width="800px">
<p><i>Overview of our proposed multimodal backdoor attack(GLA) framework. </i></p>
</div>

## 📢 News
- **[2025-11-20]** Code and visualization tools released.
<!-- - **[2025-xx-xx]** Paper accepted to [Conference Name]. -->

## 🛠️ Environment Setup

To set up the environment, we recommend using [Anaconda](https://www.anaconda.com/).

```bash
# Create a virtual environment
conda create -n gla python=3.8
conda activate gla

# Install dependencies
pip install -r requirements.txt