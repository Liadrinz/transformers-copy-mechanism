from pathlib import Path
from typing import Dict, Tuple
from argparse import ArgumentParser
from tqdm import tqdm
from rouge import Rouge

import re
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from data_utils import SummaryDataset
from modeling import BartForConditionalGenerationWithCopyMech, GPT2LMHeadModelWithCopyMech


MODELS: Dict[str, Tuple[PreTrainedTokenizer, PreTrainedModel]] = {
    "gpt2": (GPT2Tokenizer, GPT2LMHeadModel),
    "bart": (BartTokenizer, BartForConditionalGeneration),
    "gpt2-copy": (GPT2Tokenizer, GPT2LMHeadModelWithCopyMech),
    "bart-copy": (BartTokenizer, BartForConditionalGenerationWithCopyMech),
}


def train(args):
    tokenizer_cls, model_cls = MODELS[args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(args.model_name_or_path)
    collator = DataCollatorForSeq2Seq(tokenizer)
    if args.model_recover_path:
        state_dict = torch.load(args.model_recover_path)
        model.load_state_dict(state_dict)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        args.batch_size //= n_gpus
    dataset = SummaryDataset(tokenizer, args.src_file, args.tgt_file, args.model_type, args.max_src_len, args.max_tgt_len)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=1,
        local_rank=args.local_rank,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    resume = False
    if list(Path(args.output_dir).glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)


def decode(args):
    decode_out_file = f"{args.model_recover_path}.decode.txt" if args.model_recover_path else "decode.txt"
    if args.do_decode:
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        tokenizer_cls, model_cls = MODELS[args.model_type]
        tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        model = model_cls.from_pretrained(args.model_name_or_path)
        model.to(device)
        collator = DataCollatorForSeq2Seq(tokenizer)
        if args.model_recover_path:
            state_dict = torch.load(args.model_recover_path)
            model.load_state_dict(state_dict)
        dataset = SummaryDataset(tokenizer, args.src_file, args.tgt_file, args.model_type, args.max_src_len, args.max_tgt_len)
        dataloader = DataLoader(dataset, args.batch_size, collate_fn=collator)
        output_texts = []
        for batch in tqdm(dataloader):
            batch = { k: v.to(device) for k, v in batch.items() }
            input_ids = batch["input_ids"]
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_tgt_len,
                num_beams=args.beam_size,
                do_sample=args.do_sample,
                top_p=args.top_p,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                length_penalty=args.length_penalty,
                diversity_penalty=args.diversity_penalty,
                num_beam_groups=args.num_beam_groups,
                num_return_sequences=args.output_candidates,
            )
            for i in range(0, len(output), args.output_candidates):
                output_buffer = []
                for output_ids in output[i:i+args.output_candidates]:
                    output_text = tokenizer.decode(output_ids).strip()
                    output_text = output_text.split(eos_token)[-2].split(bos_token)[-1].strip()
                    output_text = re.sub(r"\s+", " ", output_text)
                    output_buffer.append(output_text)
                output_texts.append("\t".join(output_buffer))
        with open(decode_out_file, "w") as fout:
            fout.writelines([line + "\n" for line in output_texts])
    
    if args.compute_rouge:
        with open(args.tgt_file, "r") as fin:
            references = [line.strip() for line in fin]
        with open(decode_out_file, "r") as fin:
            output_texts = [line.strip() for line in fin]
        hypothesis_candidates = [text.split("\t") for text in output_texts]
        
        avg_scores = {k1: {k2: 0.0 for k2 in ["f", "r", "p"]} for k1 in ["rouge-1", "rouge-2", "rouge-l"]}
        n_cands = len(hypothesis_candidates[0])
        for hypothesis in zip(*hypothesis_candidates):
            
            # # pip install rouge-chinese
            # rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
            # scores = rouge.get_scores(hypothesis, references, avg=True)
            
            # pip install py-rouge
            rouge = Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, length_limit=False, apply_avg=True)
            scores = rouge.get_scores(list(hypothesis), references)
            
            for k1 in avg_scores:
                for k2 in avg_scores[k1]:
                    avg_scores[k1][k2] += scores[k1][k2] / n_cands
            
        print(f"ROUGE-F(1/2/l): {avg_scores['rouge-1']['f']*100:.2f}/{avg_scores['rouge-2']['f']*100:.2f}/{avg_scores['rouge-l']['f']*100:.2f}")
        print(f"ROUGE-R(1/2/l): {avg_scores['rouge-1']['r']*100:.2f}/{avg_scores['rouge-2']['r']*100:.2f}/{avg_scores['rouge-l']['r']*100:.2f}")
        print(f"ROUGE-P(1/2/l): {avg_scores['rouge-1']['p']*100:.2f}/{avg_scores['rouge-2']['p']*100:.2f}/{avg_scores['rouge-l']['p']*100:.2f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="bart")
    parser.add_argument("--model_name_or_path", type=str, default="bart-base")
    parser.add_argument("--model_recover_path", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--src_file", type=str, default="cnndm-10k/train.src")
    parser.add_argument("--tgt_file", type=str, default="cnndm-10k/train.tgt")
    parser.add_argument("--max_src_len", type=int, default=768)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    if args.task == "train":
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument("--output_dir", type=str, default="output_dir")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--num_train_epochs", type=int, default=10)
        parser.add_argument("--fp16", action="store_true")
        args = parser.parse_args()
        print(args)
        train(args)
    elif args.task == "decode":
        parser.add_argument("--beam_size", type=int, default=1)
        parser.add_argument("--do_sample", action="store_true")
        parser.add_argument("--top_p", type=float, default=None)
        parser.add_argument("--length_penalty", type=float, default=0.0)
        parser.add_argument("--diversity_penalty", type=float, default=0.0)
        parser.add_argument("--num_beam_groups", type=int, default=1)
        parser.add_argument("--output_candidates", type=int, default=1)
        parser.add_argument("--do_decode", action="store_true")
        parser.add_argument("--compute_rouge", action="store_true")
        args = parser.parse_args()
        print(args)
        decode(args)
