# Pre-training BERT Masked Language Models (MLM)

This repository contains the method to pre-train a BERT model using custom vocabulary. It was used to pre-train JuriBERT presented in [https://arxiv.org/abs/2110.01485].

It also contains the code of the classification task that was used to evaluate JuriBERT.

Our models can be found at [http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/resources#juribert] and downloaded upon request.

## Instructions
To pre-train a new BERT model you need the path to a dataset containing raw text. 
You can also specify an existing tokenizer for the model. 
Paths for saving the model and the checkpoints are required.

```python
python pretrain.py \
      --files /path/to/text \
      --model_path /path/to/save/model \
      --checkpoint /path/to/save/checkpoints \
      --epochs 30 \
      --hidden_layers 2 \
      --hidden_size 128 \
      --attention_heads 2 \
      --save_steps 10 \
      --save_limit 0 \
      --min_freq 0
```

To finetune on a classification task you need the path to the pre-trained model and a CSV file containing the classification dataset. 
You need to specify the columns containing the category and the text as well as the path for saving the final model and the checkpoints.

```python
python classification.py \
  --model "custom" \
  --pretrained_path /path/to/model.bin \
  --tokenizer_path /path/to/tokenizer.json \
  --data /path/to/data.csv \
  --category "category-column" \
  --text "text-column" \
  --model_path /path/to/save/model \
  --checkpoint /path/to/save/checkpoints 
```

You can use --help to see all the available commands.

To test the masked language model use:
```python
fill_mask = pipeline(
    "fill-mask",
    model="/path/to/model",
    tokenizer=tokenizer
)

fill_mask("Paris est la capitale de la <mask>.")
```

