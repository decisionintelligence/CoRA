# CoRA: Boosting Time Series Forecasting Foundation Models through Correlation-Aware Adapters

### Installation

1. Create virtual environment
    ```shell
    conda create -n "CoRA" python=3.10
    conda activate CoRA
    pip install -r requirements.txt
    ```

### Prepaer Datasets

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1ZrDotV98JWCSfMaQ94XXd6vh0g27GIrB/view?usp=drive_link). Create a separate folder named `./dataset` 

### Prepaer Checkpoints for Foundation Models
We provide checkpoints for the basic model used in the paper. Please download the checkpoints from [Google Drive](https://drive.google.com/file/d/1M5Opdq--wY6mOjWn6w1_CAOabnY-C_nF/view?usp=sharing). Put the checkpoints into `./ts_benchmark/baselines/LLM/checkpoints`  and `ts_benchmark/baselines/pre_train/checkpoints`.

### Train and evaluate model
- Finetuning the backbone without CoRA:

    ``` python
    python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 96, "seq_len": 384, "target_dim": 7, "is_train": 1, "sampling_rate": 0.05, "dataset": "ETTm2", "freq": "min"}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTm2/TimerModel"

    ```

- Finetuning the backbone with CoRA:

    ```python
    python ./scripts/run.py --config-path "rolling_forecast_config.json" --data-name-list "ETTm2.csv" --strategy-args '{"horizon":96}' --model-name "pre_train.TimerModel" --model-hyper-params '{"horizon": 96, "seq_len": 384, "target_dim": 7, "is_train": 1, "sampling_rate": 0.05, "dataset": "ETTm2", "freq": "min"}' --plugin-hyper-params '{"backbone_lr": 0.0001, "beta": 0.2, "dropout": 0.2, "head_dropout": 0.1, "num_after": 4, "num_before": 3, "plugin_dim": 512, "plugin_lr": 0.0001}' --adapter "Plugin_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "FEW/ETTm2/TimerModel"
    ```