import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch

from distiller_vekd import Distiller_vekd
from train import sanity_checks
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
)
from utils import git_log, init_gpu_params, logger, set_seed
from torch.utils.data import DataLoader, BatchSampler
from datasets import load_dataset
from token_add import DataCollatorForTokenAdd

MODEL_CLASSES = {
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
}




def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--force", action="store_true")

    parser.add_argument(
        "--dump_path", type=str, required=True,
    )
    parser.add_argument(
        "--data_file",type=str,required=True,)    
    parser.add_argument("--student_config", type=str, required=True,)
    parser.add_argument(
        "--student_pretrained_weights", default=None, type=str,
    )

    
    parser.add_argument("--teacher_name", type=str, required=True,)
    parser.add_argument("--token_name", type=str, required=True,)
    parser.add_argument("--temperature", default=2.0, type=float,)
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float,
    )
    parser.add_argument(
        "--alpha_mlm",
        default=0.0,
        type=float,
    )
    parser.add_argument("--alpha_mse", default=0.0, type=float,)
    parser.add_argument(
        "--alpha_cos", default=0.0, type=float,
    )
    parser.add_argument("--alpha_to", type=float, default=0.1,)
    parser.add_argument("--alpha_clm", default=0.0, type=float,)
    parser.add_argument(
        "--mlm", action="store_true",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
    )    
    

    parser.add_argument("--n_epoch", type=int, default=3,)
    parser.add_argument("--batch_size", type=int, default=5,)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=50,
    )
    parser.add_argument("--warmup_prop", default=0.05, type=float,)
    parser.add_argument("--weight_decay", default=0.0, type=float,)
    parser.add_argument("--learning_rate", default=5e-4, type=float,)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,)
    parser.add_argument("--max_grad_norm", default=5.0, type=float,)
    parser.add_argument("--initializer_range", default=0.02, type=float,)

    parser.add_argument("--n_gpu", type=int, default=1,)
    parser.add_argument("--local_rank", type=int, default=-1,)
    parser.add_argument("--seed", type=int, default=56,)

    parser.add_argument("--log_interval", type=int, default=500,)
    parser.add_argument("--checkpoint_interval", type=int, default=4000,)
    args = parser.parse_args()

    init_gpu_params(args)
    set_seed(args)
    student_config_class, student_model_class, _ = MODEL_CLASSES['distilbert']
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES['bert']

    tokenizer = teacher_tokenizer_class.from_pretrained(args.token_name)
    
    train_file=args.data_file #train corpus

    raw_datasets = load_dataset(
        'text',
        data_files=train_file,
    )
    def tokenize_function(examples):
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding= "max_length",
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,)

    if args.local_rank in [0,-1]:
        tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=["text"],
        )
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=["text"],)
    dataset=tokenized_datasets['train']
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,)
    sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    data_collator = DataCollatorForTokenAdd(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability= 0.15,
    )#masked token generation
    dataloader = DataLoader(
            tokenized_datasets['train'], collate_fn=data_collator, num_workers=1,batch_sampler=sampler,#pin_memory=True
    )

    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_pretrained_weights is not None:
        student = student_model_class.from_pretrained(args.student_pretrained_weights, config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)

    student.resize_token_embeddings(len(tokenizer.get_vocab()))
    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")

    teacher = teacher_model_class.from_pretrained(args.teacher_name, output_hidden_states=True)
    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")

    assert student.config.max_position_embeddings == teacher.config.max_position_embeddings

    torch.cuda.empty_cache()
    distiller = Distiller_vekd(
        params=args, dataloader=dataloader, student=student, teacher=teacher
    )
    distiller.train()


if __name__ == "__main__":
    main()
