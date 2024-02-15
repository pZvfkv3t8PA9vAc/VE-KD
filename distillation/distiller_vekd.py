import math
import os
import time
from distiller import Distiller
import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from utils import logger
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class Distiller_vekd(Distiller):
    def __init__(
        self, params: dict, dataloader: DataLoader, student: nn.Module, teacher: nn.Module
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        self.isdense =  (self.student.config.hidden_size != self.teacher.config.hidden_size)
        if self.isdense:
            self.dense = nn.Linear(self.student.config.hidden_size,self.teacher.config.hidden_size)
            self.dense.to(student.device)        
        self.student_config = student.config
        self.vocab_size = student.config.vocab_size
        self.dataloader=dataloader
        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        self.alpha_to = params.alpha_to
        self.mlm = params.mlm

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.multi_gpu:
            from torch.nn.parallel import DistributedDataParallel

            self.student = DistributedDataParallel(
                self.student,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                find_unused_parameters=True,
            )
            if self.isdense:
                self.dense = DistributedDataParallel(
                    self.dense,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )
        self.is_master = params.is_master
        if self.is_master:
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)

    def train(self):
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = [batch['input_ids'],batch['attention_mask'],batch['labels'],batch['labels']<30522]
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                    token_ids, attn_mask, lm_labels, token_mask = batch
                self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels,token_mask=token_mask)
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor, token_mask:torch.tensor):
        if self.mlm:
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (bs, seq_length, voc_size)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)i
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        mask0 = torch.logical_and(attention_mask,token_mask)
        mask_s = mask0.unsqueeze(-1).expand_as(s_logits)
        mask_t = mask0.unsqueeze(-1).expand_as(t_logits)
        s_logits_slct = torch.masked_select(s_logits, mask_s)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct[::,:t_logits.size(-1)]
        t_logits_slct = torch.masked_select(t_logits, mask_t)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, t_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
        loss = self.alpha_mlm * loss_mlm
        if self.alpha_ce > 0.0:
            loss_ce = (
                self.ce_loss_fct(
                    F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                    F.softmax(t_logits_slct / self.temperature, dim=-1),
                )
                * (self.temperature) ** 2
            )
            loss_ce = max((loss_ce-self.alpha_to/self.alpha_ce),loss_ce-loss_ce)
            loss += self.alpha_ce * loss_ce
        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(0)
            loss_mse = max((loss_mse-self.alpha_to/self.alpha_mse),loss_mse-loss_mse)
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)

            mask_s = mask0.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            dim_s = s_hidden_states.size(-1)
            mask_t = mask0.unsqueeze(-1).expand_as(t_hidden_states)  # (bs, seq_length, dim)
            dim_t = t_hidden_states.size(-1)


            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask_s)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim_s)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask_t)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim_t)  # (bs * seq_length, dim)

            if self.isdense:
                s_hidden_states_slct = self.dense(s_hidden_states_slct)
            assert s_hidden_states_slct.size() == t_hidden_states_slct.size()
            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)

            loss_cos = max((loss_cos-self.alpha_to/self.alpha_cos),loss_cos-loss_cos)
            loss += self.alpha_cos * loss_cos
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        if self.alpha_ce > 0.0:
            self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)
