# Imports
from transformers import BertTokenizer
import pandas
import torch
import tqdm

# Read in-domain train data
df = pandas.read_csv(
    './data/cola_public/raw/in_domain_train.tsv', 
    sep='\t',
    header=None, 
    names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Create files for feature label
train_feature_file = open('./data/processed/train.feat', 'a+')
train_label_file = open('./data/processed/train.labl', 'a+')
valid_feature_file = open('./data/processed/valid.feat', 'a+')
valid_label_file = open('./data/processed/valid.labl', 'a+')

# Create tokenizer instance
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Set sizes for the different data
train_size = int(0.9 * len(df))
val_size = len(df) - train_size

test = []

# Create label and feature files
for i in tqdm.tqdm(range(len(df))):
    
    # Check if the data is under the train creation phase
    if i <= train_size:
        # Tokenize iterate sentence
        result = tokenizer.encode_plus(
            df['sentence'][i],              # Sentence to encode.
            add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
            return_attention_mask=True,     # Construct attn. masks.
            return_tensors='pt',            # Return pytorch tensors.
        )
        
        test.append(result['attention_mask'])
        
        # Write token IDs to the feature file
        for j in result['input_ids'][0]:
            train_feature_file.write(f"{j} ")
        train_feature_file.write("\n")
        
        # Write label to the label file
        label = df['label'][i]
        train_label_file.write(f"{label}\n")
    
    # Else...
    else:
        # Tokenize iterate sentence
        result = tokenizer.encode_plus(
            df['sentence'][i],              # Sentence to encode.
            add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
            padding="max_length",
            max_length=64,
            return_attention_mask=True,     # Construct attn. masks.
            return_tensors='pt',            # Return pytorch tensors.
        )
        
        # Write token IDs to the feature file
        for j in result['input_ids'][0]:
            valid_feature_file.write(f"{j} ")
        valid_feature_file.write("\n")
        
        # Write label to the label file
        label = df['label'][i]
        valid_label_file.write(f"{label}\n")
    
print(test[:5])
input()
print(test[0])
input()
test = torch.cat(test, dim=0)
print(test)
input()
print(test.size())
input()


# Close files
train_feature_file.close()
train_label_file.close()
valid_feature_file.close()
valid_label_file.close()