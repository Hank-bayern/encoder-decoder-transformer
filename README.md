# encoder-decoder-transformer

## Getting Started

To install the necessary dependencies, run the following command:

    pip install -r requirements.txt

## Data processing
'dataset matching.py' This script responsible for macthing the original en and de dataset, and tokenizer
'dictionary and index transformer.py' This script responsible to creat the dictionary and translate tokens from words to indexs.
'expand.py' and 'shuffle.py' These are used to expand the dataset.
'testset.py' Is used to set trainingset and testset.
'mistake_index.py' and 'translation_mistake_insertion.py'Is used to get the index from output and insert into the head of testset to generate mistake insertion testset.
'tokens_classificationy.py' is used to classfy the tokens from input_vocab, output_vocab and share_vocab.

## Model
'Model-Training.ipynb' This script responsible for training the model.

## Evaluate
'test_result.py' and 'test_accuracy' Are used to evalute the model by testset.
'mistake_insertion_result.py' and 'mistake_insertion_accuracy.py' Are used to evaluate the model by mistake insertion testset.
'token_translater.py" is used to translate the result from index to words.

