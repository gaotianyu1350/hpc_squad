export TRAIN_FILE=./squad_train_text.txt
export TEST_FILE=./squad_dev_text.txt

python run_language_modeling.py \
    --output_dir=mlm_model/exp \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --warmup_steps 0 \
    --save_steps 100 \
    --fp16 \
    --logging_steps 100 \
    --learning_rate 5e-5 \
