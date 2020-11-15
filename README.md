# Task transferability
Note: This repository is a work-in-progress.

Data and code for our paper [Exploring and Predicting Transferability across NLP Tasks](https://arxiv.org/abs/2005.00770), to appear at [EMNLP 2020](https://2020.emnlp.org). 

<p align="center">
  <img src="https://github.com/tuvuumass/task-transferability/blob/main/figs/fig1.png" width="50%" alt="Task embedding pipeline.">
</p>

Figure 1: A demonstration of our task embedding pipeline. Given a target task, we first compute its task embedding and then identify the most similar source task embedding (in this example, WikiHop) from a precomputed library via cosine similarity. Finally, we perform intermediate-task transfer from the selected source task to the target task.

## Table of Contents

   * [Installation](#installation)
   * [Data](#data)
   * [Fine-tuning BERT on downstream NLP tasks](#fine-tuning-bert-on-downstream-nlp-tasks)
      * [Fine-tuning BERT on text classification/regression (CR) tasks](#fine-tuning-bert-on-text-classificationregression-cr-tasks)
      * [Fine-tuning BERT on question answering (QA) tasks](#fine-tuning-bert-on-question-answering-qa-tasks)
      * [Fine-tuning BERT on sequence labeling (SL) tasks](#fine-tuning-bert-on-sequence-labeling-sl-tasks)
   * [Intermediate-task transfer with BERT](#intermediate-task-transfer-with-bert)
      * [Intermediate-task transfer to text classification/regression (CR) tasks](#intermediate-task-transfer-to-text-classificationregression-cr-tasks)
      * [Intermediate-task transfer to question answering (QA) tasks](#intermediate-task-transfer-to-question-answering-qa-tasks)
      * [Intermediate-task transfer to sequence labeling (SL) tasks](#intermediate-task-transfer-to-sequence-labeling-sl-tasks)
   * [TextEmb](#textemb)
      * [Computing TextEmb for text classification/regression (CR) tasks](#computing-textemb-for-text-classificationregression-cr-tasks)
      * [Computing TextEmb for question answering (QA) tasks](#computing-textemb-for-question-answering-qa-tasks)
      * [Computing TextEmb for sequence labeling (SL) tasks](#computing-textemb-for-sequence-labeling-sl-tasks)      
   * [TaskEmb](#taskemb)
      * [Computing TaskEmb for text classification/regression (CR) tasks](#computing-taskemb-for-text-classificationregression-cr-tasks)
      * [Computing TaskEmb for question answering (QA) tasks](#computing-taskemb-for-question-answering-qa-tasks)
      * [Computing TaskEmb for sequence labeling (SL) tasks](#computing-taskemb-for-sequence-labeling-sl-tasks)
   * [Pretrained models and precomputed task embeddings](#pretrained-models-and-precomputed-task-embeddings) 
   * [Using task embeddings for source task selection](#using-task-embeddings-for-source-task-selection)     

## Installation
This repository is developed on Python 3.7.5, PyTorch 1.3.1 and the [Hugging Face's Transformers](https://github.com/huggingface/transformers) 2.1.1.

You should install all necessary Python packages in a [virtual environment](https://docs.python.org/3/library/venv.html). If you are unfamiliar with Python virtual environments, please check out the user guide [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Below, we create a virtual environment and activate it:
```
conda create -n task-transferability python=3.7
conda activate task-transferability
```
Now, install all necessary packages:
```
cd task-transferability
pip install -r requirements.txt
pip install [--editable] .
```

## Data
You can download the classification/regression (CR) and question answering (QA) datasets we study at this [Google Drive link](https://drive.google.com/drive/folders/1Uz6G5QuXXWgrcPHqbfxhcMPTrJjuwq0c?usp=sharing). Unfortunately, most (if not all) of the sequence labeling (SL) datasets used in our paper are not publicly available; more details can be found [here](https://github.com/nelson-liu/contextual-repr-analysis/blob/master/DATASETS.md).

## Fine-tuning BERT on downstream NLP tasks
We have separate code for fine-tuning BERT on each class of problems, including text _classification/regression_, _question answering_, and _sequence labeling_.  If you are interested in writing a common API for fine-tuning BERT on different classes of problems, check out a homework I designed for [our Advanced NLP class](https://people.cs.umass.edu/~miyyer/cs685/index.html) at this [Google Colab link](https://colab.research.google.com/drive/1K9H753cX0tD0lsoXvyHsDhrTtbnzq1bL?usp=sharing#scrollTo=7kDEdMvq9tCr) (see the function `do_target_task_finetuning()`)!

### Fine-tuning BERT on text classification/regression (CR) tasks
The following example code fine-tunes BERT on the MRPC task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/MRPC/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export TASK_NAME=MRPC
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True # set to False to freeze BERT and train the output layer only
export SAVE=all # model checkpoints to save; can be a specific checkpoint, e.g., 0, 1,.., num_train_epochs
export OUTPUT_DIR=path/to/output/dir

python ./run_finetuning_CR.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR}\
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=32  \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE} \
    --seed ${SEED_ID}
```

### Fine-tuning BERT on question answering (QA) tasks
Here is an example that fine-tunes BERT on the SQuAD-2 task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/SQuAD-2/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export VERSION_2_WITH_NEGATIVE=True
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True # set to False to freeze BERT and train the output layer only
export SAVE=all # model checkpoints to save; can be a specific checkpoint, e.g., 0, 1,.., num_train_epochs
export OUTPUT_DIR=/path/to/output/dir

python ./run_finetuning_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR} \
    --train_file ${DATA_DIR}/train.json \
    --predict_file ${DATA_DIR}/dev.json \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=12  \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE} \
    --seed ${SEED_ID}
```

### Fine-tuning BERT on sequence labeling (SL) tasks
This example code fine-tunes BERT on the NER task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/NER/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export LABEL_FILE=${DATA_DIR}/labels.txt
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True # set to False to freeze BERT and train the output layer only
export SAVE=all # model checkpoints to save; can be a specific checkpoint, e.g., 0, 1,.., num_train_epochs
export OUTPUT_DIR=/path/to/output/dir

python ./run_finetuning_SL.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --labels ${LABEL_FILE} \
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR} \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=32  \
    --learning_rate 5e-5 \
    --num_train_epochs 6 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE} \
    --seed ${SEED_ID}
```
Note: You might want to use a `Cased` model if case information is important for your task (e.g., named entity recognition or part-of-speech tagging). 

## Intermediate-task transfer with BERT
Since the target task is typically different than the source task, we will discard the output layer from the model fine-tuned on the source task and incorporate BERT with another output layer to form a task-specific model for the target task. The output layer will be learned from scratch. You can do this by running our fine-tuning code with the argument `model_load_mode = "base_model_only"`.

<p align="center">
  <img src="https://github.com/tuvuumass/task-transferability/blob/main/figs/fig2.png" width="80%" alt="Transfer results.">
</p>

Figure 2: Our experiments show that even low-data source tasks that are on the surface very different than the target task can result in transfer gains. Out-of-class transfer (e.g., from a question answering task to a classification task) succeeds in many cases, some of which are unintuitive. Here in this violin plot, each point in a violin represents the performance on the associated target task when we perform intermediate-task transfer with a particular source task and the black line corresponds to what happens when we directly fine-tune BERT on the target task without any intermediate-task transfer.

### Intermediate-task transfer to text classification/regression (CR) tasks
This example code performs intermediate-task transfer from the HotpotQA task (question answering) to the MRPC task (text classification/regression):
```bash
export SEED_ID=42
export TARGET_DATA_DIR=/path/to/MRPC/data/dir
export MODEL_TYPE=bert
export TARGET_TASK=MRPC
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True
export MODEL_LOAD_MODE=base_model_only
export SAVE=all
export SOURCE_CHECKPOINT=3
export MODEL_NAME_OR_PATH=/path/to/BERT/HotpotQA/model/checkpoint-${SOURCE_CHECKPOINT}
export OUTPUT_DIR=/path/to/output/dir

python ./run_finetuning_CR.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --task_name ${TARGET_TASK} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR} \
    --model_load_mode ${MODEL_LOAD_MODE} \
    --data_dir ${TARGET_DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=32  \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE} \
    --seed ${SEED_ID}
```

### Intermediate-task transfer to question answering (QA) tasks
The following script does intermediate-task transfer from the GParent task (sequence labeling) to the CQ task (question answering):
```bash
export SEED_ID=42
export TARGET_DATA_DIR=/path/to/CQ/data/dir
export MODEL_TYPE=bert
export VERSION_2_WITH_NEGATIVE=False
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True
export MODEL_LOAD_MODE=base_model_only
export SAVE=all
export SOURCE_CHECKPOINT=6
export MODEL_NAME_OR_PATH=/path/to/BERT/GParent/model/checkpoint-${SOURCE_CHECKPOINT}
export OUTPUT_DIR=/path/to/output/dir

python ./run_finetuning_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR} \
    --model_load_mode ${MODEL_LOAD_MODE} \
    --train_file ${TARGET_DATA_DIR}/train.json \
    --predict_file ${TARGET_DATA_DIR}/dev.json \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=12  \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE}
```

### Intermediate-task transfer to sequence labeling (SL) tasks
Below, we perform intermediate-task transfer from the MNLI task (text classification/regression) to the Conj task (sequence labeling):
```bash
export SEED_ID=42
export TARGET_DATA_DIR=/path/to/Conj/data/dir
export MODEL_TYPE=bert
export LABEL_FILE=${TARGET_DATA_DIR}/labels.txt
export CACHE_DIR=/path/to/cache/dir
export FINETUNE_FEATURE_EXTRACTOR=True
export MODEL_LOAD_MODE=base_model_only
export SAVE=all
export SOURCE_CHECKPOINT=3
export MODEL_NAME_OR_PATH=/path/to/BERT/MNLI/model/checkpoint-${SOURCE_CHECKPOINT}
export OUTPUT_DIR=/path/to/output/dir

python ./run_finetuning_SL.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --labels ${LABEL_FILE}\
    --finetune_feature_extractor ${FINETUNE_FEATURE_EXTRACTOR} \
    --model_load_mode ${MODEL_LOAD_MODE} \
    --data_dir ${TARGET_DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=32  \
    --learning_rate 5e-5 \
    --num_train_epochs 6 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --save ${SAVE}
```

## TextEmb
`TextEmb` is computed by pooling BERT’s final-layer token-level representations across an entire dataset, and as such captures properties of the text and domain. This method does not depend on the training labels and thus can be used in zero-shot transfer.

<p align="center">
  <img src="https://github.com/tuvuumass/task-transferability/blob/main/figs/fig3a.png" width="50%" alt="TextEmb space.">
</p>

Figure 3a: A visualization of the task space of `TextEmb`. It captures domain similarity (e.g., the Penn Treebank sequence labeling tasks are highly interconnected).

### Computing TextEmb for text classification/regression (CR) tasks
You can run something like the following script to compute `TextEmb` for the MRPC task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/MRPC/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export TASK_NAME=MRPC
export CACHE_DIR=/path/to/cache/dir
export OUTPUT_DIR=/path/to/output/dir

python run_textemb_CR.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --task_name ${TASK_NAME} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_train_batch_size=32  \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

Let's look at the `TextEmb` task embedding:
```python
import numpy as np

task_emb = np.load("/path/to/output/dir/avg_sequence_output.npy").reshape(-1)
print(task_emb.shape) # (768, )
print(task_emb)
```

### Computing TextEmb for question answering (QA) tasks
This example code computes `TextEmb` for the SQuAD-2 task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/SQuAD-2/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export VERSION_2_WITH_NEGATIVE=True
export CACHE_DIR=/path/to/cache/dir
export OUTPUT_DIR=/path/to/output/dir

python ./run_textemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_lower_case \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --train_file ${DATA_DIR}/train.json \
    --max_seq_length 384 \
    --per_gpu_train_batch_size=12  \
    --num_train_epochs 1 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

### Computing TextEmb for sequence labeling (SL) tasks
Here is an example that computes `TextEmb` for the NER task:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/NER/data/dir
export MODEL_TYPE=bert
export MODEL_NAME_OR_PATH=bert-base-uncased
export LABEL_FILE=${DATA_DIR}/labels.txt
export CACHE_DIR=/path/to/cache/dir
export OUTPUT_DIR=/path/to/output/dir

python ./run_textemb_SL.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_lower_case \
    --labels ${LABEL_FILE}\
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --per_gpu_train_batch_size=32  \
    --num_train_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

## TaskEmb
`TaskEmb` relies on the correlation between the fine-tuning loss function and the parameters of BERT, and encodes more information about the type of knowledge and reasoning required to solve the task. It is computationally more expensive than `TextEmb` since you need to fine-tune BERT. If you have enough training data for your task, you can try freezing BERT during fine-tuning and train the output layer only. We find empirically that `TaskEmb` computed from a fine-tuned task-specific BERT result in better correlations to task transferability in data-constrained scenarios.

<p align="center">
  <img src="https://github.com/tuvuumass/task-transferability/blob/main/figs/fig3b.png" width="50%" alt="TextEmb space.">
</p>

Figure 3b: A visualization of the task space of `TaskEmb`. It focuses more on task similarity (e.g., the two part-of-speech tagging tasks are interconnected despite their domain dissimilarity).

### Computing TaskEmb for text classification/regression (CR) tasks
In this example, we will compute `TaskEmb` for the MRPC task. We assume that you have fine-tuned BERT on MRPC already.
Use the following script to compute `TaskEmb`:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/MRPC/data/dir
export MODEL_TYPE=bert
export TASK_NAME=MRPC
export USE_LABELS=True # set to False to sample from the model's predictive distribution
export CACHE_DIR=/path/to/cache/dir
export CHECKPOINT=3
export MODEL_NAME_OR_PATH=/path/to/BERT/MRPC/model/checkpoint-${CHECKPOINT}
# we start from a fine-tuned task-specific BERT so no need for further fine-tuning
export FURTHER_FINETUNE_CLASSIFIER=False
export FURTHER_FINETUNE_FEATURE_EXTRACTOR=False
export OUTPUT_DIR=/path/to/output/dir

python ./run_taskemb_CR.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --task_name ${TASK_NAME} \
    --use_labels ${USE_LABELS} \
    --do_lower_case \
    --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
    --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --batch_size=32  \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

Let's look at the `TaskEmb` task embedding computed from a specific component of BERT, i.e., the `CLS` output vector:
```python
import numpy as np

task_emb = np.load("/path/to/output/dir/cls_output.npy").reshape(-1)
print(task_emb.shape) # (768, )
print(task_emb)
```

### Computing TaskEmb for question answering (QA) tasks
Let's compute `TaskEmb` for the SQuAD-2 task. We assume that you have fine-tuned BERT on SQuAD-2 already.
Run the following script to compute `TaskEmb`:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/SQuAD-2/data/dir
export MODEL_TYPE=bert
export VERSION_2_WITH_NEGATIVE=True
export USE_LABELS=True # set to False to sample from the model's predictive distribution
export CACHE_DIR=/path/to/cache/dir
export CHECKPOINT=3
export MODEL_NAME_OR_PATH=/path/to/BERT/SQuAD-2/model/checkpoint-${CHECKPOINT}
# we start from a fine-tuned task-specific BERT so no need for further fine-tuning
export FURTHER_FINETUNE_CLASSIFIER=False
export FURTHER_FINETUNE_FEATURE_EXTRACTOR=False
export OUTPUT_DIR=/path/to/output/dir

python ./run_taskemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --use_labels ${USE_LABELS} \
    --do_lower_case \
    --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
    --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
    --train_file ${DATA_DIR}/train.json \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --max_seq_length 384 \
    --batch_size=12  \
    --learning_rate 3e-5 \
    --num_epochs 1 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

### Computing TaskEmb for sequence labeling (SL) tasks
This example code computes `TaskEmb` for the NER task, which assumes that you have fine-tuned BERT on NER already:
```bash
export SEED_ID=42
export DATA_DIR=/path/to/NER/data/dir
export MODEL_TYPE=bert
export LABEL_FILE=${DATA_DIR}/labels.txt
export USE_LABELS=True # set to False to sample from the model's predictive distribution
export CACHE_DIR=/path/to/cache/dir
export CHECKPOINT=6
export MODEL_NAME_OR_PATH=/path/to/BERT/NER/model/checkpoint-${CHECKPOINT}
# we start from a fine-tuned task-specific BERT so no need for further fine-tuning
export FURTHER_FINETUNE_CLASSIFIER=False
export FURTHER_FINETUNE_FEATURE_EXTRACTOR=False
export OUTPUT_DIR=/path/to/output/dir

python ./run_taskemb_CR.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --labels ${LABEL_FILE} \
    --use_labels ${USE_LABELS} \
    --do_lower_case \
    --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
    --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
    --data_dir ${DATA_DIR} \
    --max_seq_length 128 \
    --batch_size=32  \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}
```

## Pretrained models and precomputed task embeddings
Coming soon...

## Using task embeddings for source task selection
Coming soon...

## Citation
If you find this repository useful, please cite our paper:
```bibtex
@inproceedings{vu-etal-2020-task-transferability,
    title = "Exploring and Predicting Transferability across {NLP} Tasks",
    author = "Vu, Tu  and
      Wang, Tong  and
      Munkhdalai, Tsendsuren  and
      Sordoni, Alessandro  and
      Trischler, Adam  and
      Mattarella-Micke, Andrew  and
      Maji, Subhransu  and
      Iyyer, Mohit",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.635",
    pages = "7882--7926",
}
```
