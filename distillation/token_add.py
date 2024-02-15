import numpy as np
import torch
from torch.utils.data import Dataset
from utils import logger
from tqdm.auto import tqdm
import transformers
from typing import Iterable, Union

from transformers import AutoTokenizer,BertJapaneseTokenizer
from transformers import DataCollatorForLanguageModeling


class LanguageModelingDataset(Dataset):
    """
    Dataset for (masked) language model task.
    Can sort sequnces for efficient padding.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: Union[
            str, transformers.tokenization_utils.PreTrainedTokenizer
        ],
        max_seq_length: int = 500,
        sort: bool = True,
        lazy: bool = False,
    ):
        """
        Args:
            texts (Iterable): Iterable object with text
            tokenizer (str or tokenizer): pre trained
                huggingface tokenizer or model name
            max_seq_length (int): max sequence length to tokenize
            sort (bool): If True then sort all sequences by length
                for efficient padding
            lazy (bool): If True then tokenize and encode sequence
                in __getitem__ method
                else will tokenize in __init__ also
                if set to true sorting is unavialible
        """
        if sort and lazy:
            raise Exception(
                "lazy is set to True so we can't sort"
                " sequences by length.\n"
                "You should set sort=False and lazy=True"
                " if you want to encode text in __get_item__ function"
            )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(
            tokenizer, transformers.tokenization_utils.PreTrainedTokenizer
        ):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer argument should be a model name"
                + " or huggingface PreTrainedTokenizer"
            )

        self.max_seq_length = max_seq_length

        self.lazy = lazy

        if lazy:
            self.texts = texts

        if not lazy:
            pbar = tqdm(texts, desc="tokenizing texts")
            self.encoded = [
                self.tokenizer.encode(text, max_length=max_seq_length)
                for text in pbar
            ]
            if sort:
                self.encoded.sort(key=len)

        self.length = len(texts)

        self._getitem_fn = (
            self._getitem_lazy if lazy else self._getitem_encoded
        )

    def __len__(self):
        """Return length of dataloader"""
        return self.length

    def _getitem_encoded(self, idx) -> torch.Tensor:
        return torch.tensor(self.encoded[idx])

    def _getitem_lazy(self, idx) -> torch.Tensor:
        encoded = self.tokenizer.encode(
            self.texts[idx], max_length=self.max_seq_length
        )

        return torch.tensor(encoded)

class DataCollatorForTokenAdd(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
#         print(special_tokens_mask)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         print(probability_matrix)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        token_indices = torch.where(inputs>=30522,1,0)
        #print(token_indices)
        masked_indices =torch.logical_or(masked_indices,token_indices)
#         print(masked_indices)
#         print(labels)
#         print(torch.logical_not(masked_indices))
        labels[torch.logical_not(masked_indices)] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)


        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(32767, labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    


class LmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.

    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """

    def __init__(self, params, data):
        self.params = params

        self.token_ids = np.array(data)
        self.lengths = np.array([len(t) for t in data])

        self.check()
        self.remove_long_sequences()
        self.remove_empty_sequences()
        self.remove_unknown_sequences()
        self.check()
        self.print_statistics()

    def __getitem__(self, index):
        return (self.token_ids[index], self.lengths[index])

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))

    def remove_long_sequences(self):
        """
        Sequences that are too long are split by chunk of max_model_input_size.
        """
        max_len = self.params.max_model_input_size
        indices = self.lengths > max_len
        logger.info(f"Splitting {sum(indices)} too long sequences.")

        def divide_chunks(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        if self.params.mlm:
            cls_id, sep_id = self.params.special_tok_ids["cls_token"], self.params.special_tok_ids["sep_token"]
        else:
            cls_id, sep_id = self.params.special_tok_ids["bos_token"], self.params.special_tok_ids["eos_token"]

        for seq_, len_ in zip(self.token_ids, self.lengths):
            assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    if sub_s[0] != cls_id:
                        sub_s = np.insert(sub_s, 0, cls_id)
                    if sub_s[-1] != sep_id:
                        sub_s = np.insert(sub_s, len(sub_s), sep_id)
                    assert len(sub_s) <= max_len
                    assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    sub_seqs.append(sub_s)

                new_tok_ids.extend(sub_seqs)
                new_lengths.extend([len(l) for l in sub_seqs])

        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tuned.
        """
        init_size = len(self)
        indices = self.lengths > 11
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} too short (<=11 tokens) sequences.")

    def remove_unknown_sequences(self):
        """
        Remove sequences with a (too) high level of unknown tokens.
        """
        if "unk_token" not in self.params.special_tok_ids:
            return
        else:
            unk_token_id = self.params.special_tok_ids["unk_token"]
        init_size = len(self)
        unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids])
        indices = (unk_occs / self.lengths) < 0.5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} sequences with a high level of unknown tokens (50%).")

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        if not self.params.is_master:
            return
        logger.info(f"{len(self)} sequences")
        # data_len = sum(self.lengths)
        # nb_unique_tokens = len(Counter(list(chain(*self.token_ids))))
        # logger.info(f'{data_len} tokens ({nb_unique_tokens} unique)')

        # unk_idx = self.params.special_tok_ids['unk_token']
        # nb_unknown = sum([(t==unk_idx).sum() for t in self.token_ids])
        # logger.info(f'{nb_unknown} unknown tokens (covering {100*nb_unknown/data_len:.2f}% of the data)')

    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t[0] for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        if self.params.mlm:
            pad_idx = self.params.special_tok_ids["pad_token"]
        else:
            pad_idx = self.params.special_tok_ids["unk_token"]
        tk_ = [list(t.astype(int)) + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)
        return tk_t, lg_t
