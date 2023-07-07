PRETRAINED_PATH="data/pretrained/deberta-v3-small/"

python -m src.run_class \
    --model_name_or_path "${PRETRAINED_PATH}" \
    --model_type roberta \
    --train_file data/squad_2.0/data_train.csv \
    --predict_file data/squad_2.0/data_test.csv \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir data/outputs/mrc_squad_2.0_classification \
    --per_gpu_eval_batch_size 12 \
    --per_gpu_train_batch_size 32 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --evaluate_during_training \
    --seed 42
