# Qlora

## Setup
```bash
conda activate qlora
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Start

 0. Copy llama-7b-hf-transformers-4.29 to localssd.
 1. Run `prepare_mmlu.py` to download mmlu data.
 2. Run `run_qlora.sh` or `run_gwqlora.sh` or `run_lora.sh`. Finetuning a llama 7b model costs about 5 hours on a A100 but the evaluation costs a lot of time too. The total running time should be within 8 hours.

! the code still has problem now !