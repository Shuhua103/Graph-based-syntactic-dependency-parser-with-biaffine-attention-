import os
# # 修改当前工作目录
# os.chdir('/home/scao/project')
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import sys
from Dataset_processing_ARC_LABEL import *
import pickle
import json
import gzip
import argparse
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#function to prepare the GLOVE embed to tune
def prepare_pretrained_embed(path_GLOVE,path_word_2_id):
    # to retrieve the original GLOVE word_vectors
    with gzip.open(path_GLOVE, 'rb') as f:
        pretrained_embeddings = pickle.load(f)

    # Make a word_2_id dictionary for GLOVE_embed looking-up
    with open(path_word_2_id, 'r') as json_file:
        word_to_idx = json.load(json_file)


    # add [PAD] to the dico and the GLOVE embedding matrix
    pad_token = "[PAD]"
    word_to_idx[pad_token] = max(word_to_idx.values()) + 1
    pad_vector = torch.zeros(1, 100)  
    pretrained_embeddings = torch.cat([pretrained_embeddings, pad_vector], dim=0)

    # calculate the mean of all embeddings vectors
    unk_vector = pretrained_embeddings.mean(dim=0, keepdim=True)
    # add [UNK] to the dico and GLOVE embeddings matrix
    unk_token = "[UNK]"
    unk_index = max(word_to_idx.values()) + 1  # Attribuer un nouvel index unique
    word_to_idx[unk_token] = unk_index

    return(torch.cat([pretrained_embeddings, unk_vector], dim=0),word_to_idx)

#Function to process the raw data into train/dev datasets for training
def prepare_dataset(path_to_corpus,path_to_corpu_dev):

    corpus_reader = SDPCorpusReader(path_to_corpus)
    corpus = corpus_reader.read_corpus()

    corpus_reader_dev = SDPCorpusReader(path_to_corpu_dev)
    corpus_dev = corpus_reader_dev.read_corpus()

    # Print corpus summary stats
    print(corpus.summary())
    print(corpus_dev.summary())

    # Access corpus examples directly from corpus
    examples = corpus.examples
    examples_dev = corpus_dev.examples

    data_processor = DataProcessor(enable_labels=True)
    X_train, Y_train = data_processor.get_X_y_train(examples)
    X_dev, Y_dev = data_processor.get_X_y_train(examples_dev)

    return(X_train, Y_train,X_dev, Y_dev)



# Auxilliary function for removing the unwanted embed from Bert output and repad
def remove_and_pad(embeddings, mask, max_length):
    new_embeddings = []

    for i in range(embeddings.size(0)):
        valid_embeds = embeddings[i][mask[i].any(dim=-1)]  
        num_valid = valid_embeds.size(0)
        if num_valid > max_length:
            valid_embeds = valid_embeds[:max_length]  
            num_valid = max_length
        num_padding = max_length - num_valid
        padding = torch.zeros((num_padding, embeddings.size(2)), device=embeddings.device)
        new_embed = torch.cat((valid_embeds, padding), dim=0)
        new_embeddings.append(new_embed)

    return torch.stack(new_embeddings)


#Auxilliary function to create the mask to ignore PAD tokens in loss computation or performance evaluation
def mask_to_ignore_pad(max_length,batch_lengths,pred_logits):

    seq_range = torch.arange(max_length, device=pred_logits.device).unsqueeze(0)  # (1, max_length)
    lengths_expanded = batch_lengths.unsqueeze(-1)  # (batch_size, 1)
    mask = seq_range < lengths_expanded  # （batch_size, max_length）
    pad_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  #（batch_size, max_length, max_length]

    return(pad_mask)



# 2 functions to calculate losses
def loss_arc(logits,y_arc,criterion,pad_mask):
    #Loss computation for ARC
    losses = criterion(logits, y_arc.to(device)) # out:(batch_size,max_length,max_length)
    #apply the pad_mask to ignore pad-relatad cells of loss
    losses = losses * pad_mask #out:(batch_size,max_length,max_length)
    #calculate the average loss of 1 batch,considering only non-pad valid losses
    loss = losses.sum() / pad_mask.sum() #out: scalar
    return(loss)

def loss_for_label(logits_label,y_label,criterion_label,pad_mask):  
    #Loss computation for LABEL
    logits_label = logits_label.reshape(-1, logits_label.size(-1))  # (batch_size * max_length * max_length, num_labels)
    y_label = y_label.view(-1).to(device) # (batch_size * max_length * max_length)
    pad_mask_flattened = pad_mask.reshape(-1)

    logits_label = logits_label[pad_mask_flattened]  # Shape: (num_non_pad, num_labels)
    y_label = y_label[pad_mask_flattened]  # Shape: (num_non_pad)
    losses_label = criterion_label(logits_label, y_label.long().to(device))
    loss_label = losses_label.sum() / pad_mask.sum() #out: scalar

    return(loss_label)


#The function to perform training and evaluation on dev-set in each epoch
def train_and_dev(max_epochs,model,learning_rate,max_length,tokenizer,word_to_idx,X_train,Y_train,X_dev,Y_dev):
    
    list_F1=[]
    list_F1_label=[]
    list_train_loss=[]
    list_dev_loss=[]
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.9))
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    criterion_label = nn.CrossEntropyLoss(ignore_index=0)

    best_f1=float(0) #we choose the best_forming model according to the F1 on arc prediction
    for epoch in range(0,max_epochs):
        print('EPOCH NUMBER',epoch,'\n')

        #Switch to traning mode to ensure the normal function of Dropout/BatchNorm
        model.train()
        #Get the batches of examples(shuffled at every epoch)
        train_batches = Batch.create_batches(X_train, Y_train, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True,enable_labels=True)
        # train_batches=train_batches_toy
        loss_train_batch = 0
        nb_example_batch=0
        for (ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,y_train_label,batch_lengths,y_train_arc) in train_batches:

            ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks = [tensor.to(device) for tensor in [ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks]]

            #Forward pass to get the SDP score matrix
            logits, logits_label = model(ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,batch_lengths,max_length) #out:(batch_size,max_length,max_length)
            #Create a pad_mask matrix of boolean to indicate the validity of cells(if or not concerning [pad])
            pad_mask=mask_to_ignore_pad(max_length+1,batch_lengths,logits)

            #Loss computation for ARC
            loss=loss_arc(logits,y_train_arc,criterion,pad_mask)
            #Loss computation for LABEL
            loss_label=loss_for_label(logits_label,y_train_label,criterion_label,pad_mask)
            #Joint Loss of loss_arc and loss_label
            loss = loss + loss_label
            #Upadate parameters according to the batch loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train_batch += loss * pad_mask.sum()
            nb_example_batch += pad_mask.sum()
        
        average_loss = loss_train_batch / nb_example_batch
        list_train_loss.append(average_loss)
        print("loss_averaged_train", average_loss)

        #For early-stopping on dev set (metrics: F1 socre):
        model.eval()   
        loss_dev_batch = 0
        nb_example_batch=0
        all_preds = []
        all_gold = []
        all_preds_label = []
        all_gold_label = []
        with torch.no_grad():
            dev_batches = Batch.create_batches(X_dev, Y_dev, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True,enable_labels=True)

            for (ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,y_dev_label,batch_lengths,y_dev_arc) in dev_batches:

                ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks = [tensor.to(device) for tensor in [ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks]]

                #Use current model to predict
                pred_logits, pred_logits_label = model(ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,batch_lengths,max_length) #out:(batch_size,max_length,max_length)
                #Arc predictions transformed into class-id
                pred = (torch.sigmoid(pred_logits)> 0.5).int()
                #Label predictions transformed into class-id
                softmax_probs = torch.softmax(pred_logits_label, dim=-1)  
                pred_label = torch.argmax(softmax_probs, dim=-1) 

                pad_mask = mask_to_ignore_pad(max_length+1,batch_lengths,pred_logits)  # (batch_size, max_length, max_length)

                #Loss computation for ARC
                loss=loss_arc(pred_logits,y_dev_arc,criterion,pad_mask)        
                #Loss computation for LABEL
                loss_label=loss_for_label(pred_logits_label,y_dev_label,criterion_label,pad_mask)         
                #Joint Loss of loss_arc and loss_label
                loss = loss + loss_label
                loss_dev_batch += loss * pad_mask.sum()
                nb_example_batch += pad_mask.sum()

                #Arc score
                pred = pred[pad_mask]
                y_dev_arc = y_dev_arc.to(device)[pad_mask].int()
                all_preds.append(pred.cpu().numpy())
                all_gold.append(y_dev_arc.cpu().numpy())

                #Label score
                pred_label = pred_label[pad_mask]
                y_dev_label = y_dev_label.to(device)[pad_mask].int()
                all_preds_label.append(pred_label.cpu().numpy())
                all_gold_label.append(y_dev_label.cpu().numpy())
                

            #F1 score for arc existence
            f1_current = f1_score(np.concatenate(all_preds), np.concatenate(all_gold), average='binary')
            list_F1.append(f1_current)
            print("f1-arc",f1_current)

            #F1 socre for labels
            all_preds_label = np.concatenate(all_preds_label)
            all_gold_label = np.concatenate(all_gold_label)
            #Make a mask to find the ids of non-zero gold labels cells (i.e. there does exist an edge)
            non_zero_indices = all_gold_label != 0
            # Use the mask to filter valid predictions/gold_labels
            filtered_preds = all_preds_label[non_zero_indices]
            filtered_gold = all_gold_label[non_zero_indices]
            # calculate F1-score just for valid cells
            f1_current_label = f1_score(filtered_preds, filtered_gold, average='weighted')

            # f1_current_label = f1_score(np.concatenate(all_preds_label), np.concatenate(all_gold_label), average='weighted')
            list_F1_label.append(f1_current_label)
            print("f1-label",f1_current_label)

            average_loss_dev = loss_train_batch / nb_example_batch
            list_dev_loss.append(average_loss_dev)
            print("loss_averaged_dev", average_loss_dev)
            
            #save the best-performing model obtained in 
            if f1_current > best_f1:
                best_f1 = f1_current
                model_save_path = "your_model_trained.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"New best F1 {best_f1} at epoch {epoch}, model saved.")

        
                list_F1 = [tensor.tolist() for tensor in list_F1]
                list_F1_label = [tensor.tolist() for tensor in list_F1_label]
                list_train_loss = [tensor.tolist() for tensor in list_train_loss]
                list_dev_loss = [tensor.tolist() for tensor in list_dev_loss]

                data = {
                    "list_F1": list_F1,
                    "list_train_loss": list_train_loss,
                    "list_dev_loss": list_dev_loss,
                    "list_F1_label": list_F1_label
                }
                with open("train_log.json", "w") as json_file:
                    json.dump(data, json_file, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument('-train', '--path_to_corpus', default=None, help='training dataset')
parser.add_argument('-dev', '--path_to_corpus_dev', default=None, help='dev dataset')  
parser.add_argument('-lr', '--learning_rate', type=float, default=None, help='learning rate')  
parser.add_argument('-epo', '--max_epochs', type=int, default=None, help='max epochs')  
parser.add_argument('-bert', '--if_freeze_bert', type=bool, default=False, help='if freeze bert') 

args = parser.parse_args()

path_to_corpus = args.path_to_corpus
path_to_corpus_dev = args.path_to_corpus_dev  
max_length = 200

# Hyperparameters/args to parse from command line:
learning_rate = args.learning_rate
max_epochs = args.max_epochs
if_freeze_bert = args.if_freeze_bert

# Prepare the Glove embedding to tune
pretrained_embeddings, word_to_idx = prepare_pretrained_embed('word_vectors_GLOVE_original.pkl.gz', 'word_to_idx_original.json')

# Prepare the training/dev datasets and create batches
X_train, Y_train, X_dev, Y_dev = prepare_dataset(path_to_corpus, path_to_corpus_dev)

from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create training batches
batches_train = Batch.create_batches(X_train, Y_train, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True, enable_labels=True)
batches_dev = Batch.create_batches(X_dev, Y_dev, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True, enable_labels=True)

print(f'Batches number(train): {len(batches_train)}')
print(f'Batches number(dev): {len(batches_dev)}')

from SDP_parser_arc_label import *

# Instantiation of the model (SDP_scorer)
batch_size = 8
embedding_dim = 868
vocab_size = len(word_to_idx)
num_labels = 31
hidden_dim_LSTM = 600
output_size_MLP = 600
MLP_hidden_layer_size = 600

model = SDP_scorer_arc_label(
    embedding_dim, hidden_dim_LSTM, vocab_size, output_size_MLP, MLP_hidden_layer_size, 
    pretrained_embeddings, num_labels, if_freeze_bert=if_freeze_bert  # 使用解析后的参数值
).to(device)

# Training
train_and_dev(max_epochs, model, learning_rate, max_length, tokenizer, word_to_idx, X_train, Y_train, X_dev, Y_dev)
