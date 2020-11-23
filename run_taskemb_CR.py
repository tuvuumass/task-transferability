# coding=utf-8
""" Compute TaskEmb for classification/regression tasks."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Subset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,
                          BertConfig_TaskEmbeddings as BertConfig,
                          BertForSequenceClassification_TaskEmbeddings as BertForSequenceClassification,
                          BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_Fisher(args, model, input_mask, total_tokens):
    outputs = {}

    base_model = model.bert
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad if args.feature_type == 'grads' else parameter
            if score is not None and name not in outputs:
                score = score ** args.pow
                outputs[name] = score
    # activations
    for key in ['multihead_output', 'layer_output']:
        model_outputs = base_model._get_model_outputs(key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = 'encoder.layer.{}.{}'.format(i, key)
            model_outputs_i = model_outputs[i].grad if args.feature_type == 'grads' else model_outputs[i]

            if model_outputs_i is not None:
                score = torch.einsum("ijk,ij->ijk", [model_outputs_i,   # batch_size x max_seq_length x hidden_size
                                                 input_mask.float()])   # batch_size x max_seq_length
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score ** args.pow
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score
    # cls output
    name = 'cls_output'
    score = base_model._get_model_outputs(key=name).grad if args.feature_type == 'grads' else base_model._get_model_outputs(key=name) # batch_size x hidden_size

    if score is not None and name not in outputs:
        score = score.sum(0)
        score = score ** args.pow
        # normalize
        score = score / total_tokens
        outputs[name] = score

    # task-specific layer
    for name, parameter in model.named_parameters():
        if args.model_type not in name:
            score = parameter.grad if args.feature_type == 'grads' else parameter
            if score is not None and name not in outputs:
                score = score ** args.pow
                outputs[name] = score

    return outputs


def compute_Fisher_no_labels(args, model, input_mask, logits):
    total_tokens = input_mask.float().detach().sum().data

    if args.num_labels == 1:
        #  We are doing regression
        if args.num_softmax_classifiers > 1:
            raise ValueError("Not implemented.")
        else:
            normal = Normal(logits, scale=torch.tensor([args.FIM_scale]).to(logits.device))
            num_trials = 0
            sampled_logits = None
            while num_trials < args.num_trials_for_FIM:
                sample = normal.sample(sample_shape=(1,)).squeeze(0)
                sample.to(logits.device)
                if ((sample >= 0.0) & (sample <= 5.0)).sum().item() == logits.size(0):
                    num_trials += 1
                    if sampled_logits is None:
                        sampled_logits = sample
                    else:
                        sampled_logits = torch.cat((sampled_logits, sample), dim=1)
            sampled_logits = -((sampled_logits - logits) ** 2) / (2 * (args.FIM_scale ** 2))
            sampled_logits = sampled_logits.sum(0).sum(0) / sampled_logits.numel()

            model.zero_grad()
            if args.finetune_classifier:
                sampled_logits.backward(retain_graph=True)
            else:
                sampled_logits.backward()
            outputs = compute_Fisher(args, model, input_mask, total_tokens)
    else:
        #  We are doing classification
        if args.num_softmax_classifiers > 1:
            raise ValueError("Not implemented.")
        else:
            softmax_logits = torch.softmax(logits, dim=1) # batch_size x num_labels
            sampled_indices = torch.multinomial(softmax_logits, args.num_trials_for_FIM, True)
            log_softmax_logits = torch.log(softmax_logits)
            sampled_log_softmax_logits = torch.gather(log_softmax_logits, dim=1, index=sampled_indices)

            sampled_log_softmax_logits = sampled_log_softmax_logits.sum(0).sum(0) / sampled_log_softmax_logits.numel()

            model.zero_grad()
            if args.finetune_classifier:
                sampled_log_softmax_logits.backward(retain_graph=True)
            else:
                sampled_log_softmax_logits.backward()
            outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_Fisher_with_labels(args, model, input_mask, loss):
    total_tokens = input_mask.float().detach().sum().data

    model.zero_grad()
    loss.backward()
    outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_taskemb(args, train_dataset, model):
    """ Feed task data through the model """
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs

    if args.finetune_feature_extractor and not args.finetune_classifier:
        raise ValueError("finetune_classifier should be True when finetune_feature_extractor is True.")

    if args.finetune_classifier:
        model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']

        if args.finetune_feature_extractor:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                            and args.model_type not in n],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                            and args.model_type not in n],
                 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

    else:
        model.eval()
        optimizer = None
        scheduler = None

    logger.info("***** Compute TaskEmb *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Batch size = %d", args.batch_size)

    total_num_examples = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    global_feature_dict = {}
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            outputs = model(**inputs)

            loss, logits = outputs[0], outputs[1]

            input_mask = inputs['attention_mask']

            if not args.use_labels:
                feature_dict = compute_Fisher_no_labels(args, model, input_mask, logits)
            else:
                feature_dict = compute_Fisher_with_labels(args, model, input_mask, loss)
            ###
            if len(global_feature_dict) == 0:
                for key in feature_dict:
                    global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
            else:
                for key in feature_dict:
                    global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()

            if ((not args.use_labels) and args.finetune_classifier):
                model.zero_grad()
                loss.backward()

            if args.finetune_classifier:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()

            model.zero_grad()
            num_examples += inputs['input_ids'].size(0)
        total_num_examples += num_examples

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    # Save features
    for key in global_feature_dict:
        np.save(os.path.join(args.output_dir, '{}.npy'.format(key)), global_feature_dict[key])


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Task embeddings
    parser.add_argument("--num_softmax_classifiers", default=1, type=int,
                        help="Number of softmax classifiers on top of Bert's output.")
    parser.add_argument("--pow", type=float, default=2.0,
                        help="Return features to the power pow.")
    parser.add_argument("--feature_type", default='grads', type=str,
                        help="The type of the features selected in ['grads', 'weights']")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--retain_gradients", default=True, type=eval,
                        help="Whether to retain gradients at each layer output of the feature extractor.")
    parser.add_argument("--do_pooling", default=True, type=eval,
                        help="Whether to pool the feature extractor.")
    parser.add_argument("--use_labels", default=True, type=eval,
                        help="Whether to use training labels or sample from the model's predictive distribution \n"
                             "pθ(y|xn), e.g., to compute the theoretical Fisher information.")
    parser.add_argument("--num_trials_for_FIM", type=int, default=100,
                        help="Number of trials to sample from the model's predictive distribution pθ(y|xn).")
    parser.add_argument("--FIM_scale", type=float, default=0.25,
                        help="Standard deviation of the distribution used to compute the theoretical FIM.")
    parser.add_argument("--finetune_classifier", default=False, type=eval,
                        help="Whether to fine-tune the final classifier.")
    parser.add_argument("--finetune_feature_extractor", default=False, type=eval,
                        help="Whether to fine-tune the feature extractor.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--data_subset", type=int, default=-1,
                        help="If > 0: limit the data to a subset of data_subset instances.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--save', type=str, default='all',
                        help="Select load mode from ['all', '0', '1', '2', '3', ...]")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'run_args.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=2))
        f.close()

    # Setup CUDA, GPU training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.n_gpu > 1:
        raise ValueError("This code only supports a single GPU.")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=args.num_labels,
                                          finetuning_task=args.task_name,
                                          num_softmax_classifiers=args.num_softmax_classifiers,
                                          retain_gradients=args.retain_gradients,
                                          do_pooling=args.do_pooling,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model.to(args.device)

    logger.info("List of model named parameters:")
    for n, p in list(model.named_parameters()):
        logger.info("%s", n)
    logger.info("Training/evaluation parameters %s", args)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    tokenizer.save_pretrained(args.output_dir)

    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    if args.data_subset > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.data_subset, len(train_dataset)))))
    compute_taskemb(args, train_dataset, model)


if __name__ == "__main__":
    main()
