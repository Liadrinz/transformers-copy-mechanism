from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BartTokenizer, GPT2Tokenizer


class SummaryDataset(Dataset):
    
    def __init__(self, tokenizer: PreTrainedTokenizer, src_file, tgt_file, model_type="bart", max_src_len=768, max_tgt_len=256) -> None:
        super().__init__()
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        with open(src_file) as fsrc, open(tgt_file) as ftgt:
            self.srcs = [line.strip() for line in fsrc]
            self.tgts = [line.strip() for line in ftgt]
        assert len(self.srcs) == len(self.tgts)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __getitem__(self, idx):
        src = self.srcs[idx]
        tgt = self.tgts[idx]
        if self.model_type.startswith("bart"):
            src_ids = self.tokenizer.encode(
                src, add_special_tokens=False, truncation=True, max_length=self.max_src_len-2)
            tgt_ids = self.tokenizer.encode(
                tgt, add_special_tokens=False, truncation=True, max_length=self.max_tgt_len-1)
            input_ids = [self.bos_id] + src_ids + [self.eos_id]
            attention_mask = [1] * len(input_ids)
            labels = tgt_ids + [self.eos_id]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        elif self.model_type.startswith("gpt"):
            src_ids = self.tokenizer.encode(
                src, add_special_tokens=False, truncation=True, max_length=self.max_src_len-2)
            tgt_ids = self.tokenizer.encode(
                tgt, add_special_tokens=False, truncation=True, max_length=self.max_tgt_len-1)
            input_ids = src_ids + [self.bos_id] + tgt_ids + [self.eos_id]
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(src_ids) + [1] + [1] * len(tgt_ids) + [1]
            labels = [-100] * len(src_ids) + [-100] + tgt_ids + [self.eos_id]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels
            }
        else:
            raise ValueError(f"Unexpected model type: `{self.model_type}`")
    
    def __len__(self):
        return len(self.srcs)


def test_gpt_dataset():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SummaryDataset(
        tokenizer, "cnndm-10k/training.article.10k", "cnndm-10k/training.summary.10k", model_type="gpt2")
    collator = DataCollatorForSeq2Seq(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for batch in dataloader:
        print(batch)
        break


def test_bart_dataset():
    tokenizer = BartTokenizer.from_pretrained("bart-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SummaryDataset(
        tokenizer, "cnndm-10k/training.article.10k", "cnndm-10k/training.summary.10k", model_type="bart")
    collator = DataCollatorForSeq2Seq(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    test_gpt_dataset()
    test_bart_dataset()