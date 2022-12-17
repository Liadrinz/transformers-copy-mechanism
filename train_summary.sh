export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

MODEL_TYPE=bart-copy
MODEL_NAME=fnlp/bart-base
OUTPUT_DIR=output_dir/${MODEL_TYPE}/

python3 -m torch.distributed.launch --nproc_per_node 4 run_summary.py train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size 8 \
    --src_file cnndm-10k/training.article.10k \
    --tgt_file cnndm-10k/training.summary.10k \
    --max_src_len 768 \
    --max_tgt_len 256 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps 4 \
    --lr 0.00003 \
    --num_train_epochs 10 \
    --fp16
