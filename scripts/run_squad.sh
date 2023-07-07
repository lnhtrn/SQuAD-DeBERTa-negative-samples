PRETRAINED_PATH="data/pretrained/deberta-v3-small/"
TRAIN_PATH="data/squad_2.0/squad_train-v2.0.json"
TEST_PATH="data/squad_2.0/squad_dev-v2.0.json"
OUTPUT_DIR="data/outputs/mrc_vi_squad_2.0"
TRAINING_STRAT="original"

python -m src.run_squad \
    --model_name_or_path "${PRETRAINED_PATH}" \
    --model_type roberta \
    --train_file "${TRAIN_PATH}" \
    --predict_file "${TEST_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --training_strategy "${TRAINING_STRAT}" \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --ratio "0.7,0,3" \
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
