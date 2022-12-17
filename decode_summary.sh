export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

MODEL_TYPE=bart-copy
MODEL_NAME=fnlp/bart-base
CKPT_STEP=3120
OUTPUT_DIR=output_dir/${MODEL_TYPE}/
MODEL_RECOVER_PATH=${OUTPUT_DIR}/checkpoint-${CKPT_STEP}/pytorch_model.bin

python3 -u run_summary.py decode \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_recover_path ${MODEL_RECOVER_PATH} \
    --batch_size 8 \
    --src_file cnndm-10k/test.article.500 \
    --tgt_file cnndm-10k/test.summary.500 \
    --max_src_len 768 \
    --max_tgt_len 256 \
    --seed 42 \
    --beam_size 2 \
    --do_decode \
    --compute_rouge
