import torch

from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.bart.tokenization_bart import BartTokenizer
from modeling import GPT2LMHeadModelWithCopyMech, BartForConditionalGenerationWithCopyMech


def test_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModelWithCopyMech.from_pretrained("gpt2")
    
    src_input_ids = tokenizer.encode("What is your name ?")
    tgt_input_ids = tokenizer.encode(" My name is Thomas .")
    input_ids = src_input_ids + tgt_input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(src_input_ids) + tgt_input_ids
    
    # training
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    labels = torch.tensor([labels])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(outputs.logits)
    
    # inference
    src_input_ids = torch.tensor([src_input_ids])
    output_ids = model.generate(input_ids=src_input_ids, src_ids_for_copy=src_input_ids, num_beams=4)
    print(tokenizer.batch_decode(output_ids))


def test_bart():
    tokenizer = BartTokenizer.from_pretrained("bart-base")
    model = BartForConditionalGenerationWithCopyMech.from_pretrained("bart-base")
    
    src_input_ids = tokenizer.encode("<s> What is your name ? </s>")
    tgt_input_ids = tokenizer.encode("<s> My name is Thomas . </s>")
    
    # training
    input_ids = torch.tensor([src_input_ids])
    attention_mask = torch.tensor([[1] * len(src_input_ids)])
    labels = torch.tensor([tgt_input_ids])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(outputs.logits)
    
    # inference
    output_ids = model.generate(input_ids=input_ids, src_ids_for_copy=input_ids, num_beams=4)
    print(tokenizer.batch_decode(output_ids))


if __name__ == "__main__":
    # test_gpt2()
    test_bart()
