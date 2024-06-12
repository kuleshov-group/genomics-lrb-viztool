# Analyzing results of Genomics LRB Benchmarking

Use the notebook [`lrb_results_analysis.ipynb`](lrb_results_analysis.ipynb) to analyze results of the initial model benchmarking; slice results by distances to transcription start sites (TSS) and different annotations (e.g., promoter, enhancer, intron/exon, etc.).

## Getting started
**Step 1:** Clone this repository:
```bash
git clone git@github.com:kuleshov-group/genomics-lrb-viztool.git
cd genomics-lrb-viztool
```

**Step 2:** Create conda env and install requirements:
```bash
conda create -n lrb_env python=3.10
conda activate lrb_env
pip install -r requirements.txt
```

**Step 3:** Download and unzip [results files](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/blob/main/results_with_annotations.zip) from the HF dataset:
```bash
wget -O results_with_annotations.zip https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/resolve/main/results_with_annotations.zip?download=true
unzip results_with_annotations.zip
```

Start the notebook server and use this notebook [`lrb_results_analysis.ipynb`](lrb_results_analysis.ipynb):
```bash
jupyter notebook
```
Run all cells.
Use the widgets to analyze different tasks by various splits.
![notebook-usage](assets/lrb_analysis.gif)

