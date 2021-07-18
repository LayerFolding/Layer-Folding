# Layer-Folding: Folded MobilenetV2
Use this code to run and evaluate folded mobilenetv2 with depth multiplier 0.75, 1, 1.4

## Requirements
To install requirements:
```
pip install -r requirements.txt
```
Also, download and extract the checkpoints from the following link:

## Run Evaluation
In order to run the evaluation use the main script:

```
python main.py --dm 1.0 --dataset-dir PATH_TO_IMAGENET --checkpoint-dir PATH_TO_DOWNLOADED_CHECKPOINT_DIR
```
