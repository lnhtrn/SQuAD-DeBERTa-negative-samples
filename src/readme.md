The scripts to train and evaluate the models can be found in [/scripts](scripts).

---

### 1. [Generate negative samples](src/sample_replace_generate.py) 

To generate negative samples, the samples must first go through preprocessing: `ner` for *replace method*, and `token-classification` for *swap method*.

```python
# get preprocessing pipeline 
if args.generate_type=='replace':
    classifier = pipeline("ner", aggregation_strategy='average', device=0)
    print("Downloaded ner pipeline.")
else:
    classifier = pipeline("token-classification", model = "vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple", device=0)
    print("Downloaded token-classification pipeline.")
    with open('data/squad_2.0/squad_labelled_types.json') as file: 
        all_types = json.load(file)
    with open('data/squad_2.0/id_labelled.json') as file:
        label_dict = json.load(file)
```

This preprocessing will generate the ner or token context for each sample, and new questions will be generated based on the context.

The [script](scripts/run_generate.sh) is set up with these parameters:

    --ratio 2 \
    --file_path "${FILE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --generate_type "${GENERATE_TYPE}" \
    --context_cache

The ratio is the 1/n of new negative samples generated based on the positive samples. `--context_cache` is an optional `bool`, which equals to `true` when the context was already generated for the task.



### 2. [Classification model](src/run_class.py) 

This model is trained on [DeBERTa v3 small](https://huggingface.co/microsoft/deberta-v3-small) MRC/Question-Answering pretrained model by Microsoft. 

The last layer was adapted to a classification model, which trained on the labeled data indicated in the [readme](readme.md) file of the project. 

```python
LABEL_LIST = ['What', 'Count', 'When', 'Why', 'Who', 'How', 'Where']
LABEL2INT = {LABEL_LIST[i]: i for i in range(len(LABEL_LIST))}
INT2LABEL = {i: LABEL_LIST[i] for i in range(len(LABEL_LIST))}
TOK_MAP = {339: 'what', 328: 'who', 399: 'where', 579: 'why', 335: 'when', 319: 'which', 361: 'how'}
```

These variables are set up based on the different question types in the labeled data. These are used to convert the pretrained model to a Sequence Classification model:

```python
# add num labels to config 
config.num_labels = len(LABEL_LIST)
print("Number of labels are", len(LABEL_LIST))

# converting model to classification model 
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
    ignore_mismatched_sizes=True
)
```

During training, the label weights are also computed to ensure correct loss:

```python
# compute label weights 
class_weights = [0]*len(LABEL_LIST)
for lb in batch['labels']:
    class_weights[lb] += 1
class_weights = torch.tensor(class_weights,dtype=torch.float).to(args.device)

```

The [script](scripts/run_class.sh) is set up with these parameters:

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

The parameters are set up as usual.



### 3. [Question-Answering model](src/run_squad.py)

This model is trained on [DeBERTa v3 small](https://huggingface.co/microsoft/deberta-v3-small) MRC/Question-Answering pretrained model by Microsoft. 

The model is trained with maximum 50000 samples for each epoch by using `DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler`. 

The [script](scripts/run_squad.sh) is set up with these parameters:

    --model_name_or_path "${PRETRAINED_PATH}" \
    --model_type roberta \
    --train_file "${TRAIN_PATH}" \
    --predict_file "${TEST_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --training_strategy "${TRAINING_STRAT}" \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --ratio "0.6,0,4" \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_gpu_eval_batch_size 16 \
    --per_gpu_train_batch_size 32 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --evaluate_during_training \
    --seed 42

The `--ratio` parameter indicates 0.6 negative samples and 0.4 positive samples. Other parameters are set up as usual.
