# FengWu: Pushing Skillful Global Weather Forecasts beyond 10 Days Lead

This repository presents the inference code and pre-trained model of FengWu, a deep learning-based weather forecasting model that pushes the skillful global weather forecasts beyond 10 days lead. The original version of FengWu has 37 vertical levels. To make it easier for real-time evaluation with operational analysis data, the pre-trained model released here accepts 13 vertical levels.   
If you are interested in the technique details, please refer to the arxiv version: https://arxiv.org/abs/2304.02948. 

If you have any questions, feel free to contact Dr. Lei Bai <bailei@pjlab.org.cn>. 

## Requirements

```
pip install -r requirement_gpu.txt
```

## Files

```plain
├── root
│   ├── input_data
│   │   ├── input1.npy
│   │   ├── input2.npy
│   ├── output_data
│   ├── fengwu_v1.onnx
│   ├── fengwu_v2.onnx
│   ├── inference.py
│   ├── data_mean.npy
│   ├── data_std.npy
```

## Downloading trained models

Fengwu without transfer learning (fengwu_v1.onnx): [Onedrive(https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EVA6V_Qkp6JHgXwAKxXIzDsBPIddo5RgDtGCBQ-sQbMmwg)]


Fengwu with transfer learning (fengwu_v2.onnx, finetune the model with analysis data up to 2021): [Onedrive(https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/EZkFM7nQcEtBve6MsqlWaeIB_lmpa__hX0I8QYOPzf-X6A)]


## Data Format

The model takes two consecutive six-hour data frames as input. input1.npy represents the atmospheric data at the first time moment, while input2.npy represents the atmospheric data six hours later. For example, if input1.npy represents the atmospheric state at 6:00 AM on January 1, 2018, then input2.npy represents the atmospheric state at 12:00 PM on the same day. The first predicted data corresponds to the atmospheric state at 6:00 PM on January 1, 2018, and the second predicted data corresponds to the atmospheric state at 12:00 AM on January 2, 2018.

The data is organized in the following order: Each individual data has a shape of 69x721x1440, where 69 represents 69 atmospheric features. The  latitude range is the [90N, 90S], and the longitude range is [0, 360]. The first four variables are surface variables in the order of ['u10', 'v10', 't2m', 'msl'], followed by non-surface variables in the order of ['z', 'q', 'u', 'v', 't']. Each data has 13 levels, which are ordered as [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]. Therefore, the order of the 69 variables is [u10, v10, t2m, msl, z50, z100, ..., z1000, q50, q100, ..., q1000, t50, t100, ..., t1000].

Data instance download address： (data): [https://drive.google.com/drive/folders/11i_l-mEQ7K5OcfbZd9jeBpfr_BGen9M0?usp=drive_link]

