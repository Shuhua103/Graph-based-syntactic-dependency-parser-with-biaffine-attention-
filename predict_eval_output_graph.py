import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
import sys
from Dataset_processing_ARC_LABEL import *
from SDP_parser_arc_label import *
from output_batch_graphs_ARC_LABEL import *
import argparse
import sys
import pickle
import json
import gzip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#function to prepare the GLOVE embed to tune
def prepare_GLOVE_dico(path_word_2_id):
    with open(path_word_2_id, 'r') as json_file:
        word_to_idx = json.load(json_file)

    pad_token = "[PAD]"
    word_to_idx[pad_token] = max(word_to_idx.values()) + 1
    unk_token = "[UNK]"
    unk_index = max(word_to_idx.values()) + 1  # Attribuer un nouvel index unique
    word_to_idx[unk_token] = unk_index
    return(word_to_idx)

    

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

#Function to use the loadeded model to predict
def predict(batches,model,max_length):
    model.eval()   
    nb_example_batch=0
    
    all_preds = []
    all_gold = []
    all_preds_label =[]
    all_gold_label = []
    pred_matrices_arc = []
    pred_matrices_label = []
    sentences = []
    with torch.no_grad():
        for batch  in batches:
            (ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,y_dev_label,batch_lengths,y_dev_arc) = batch

            ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks = [tensor.to(device) for tensor in [ids_glove_batch, input_ids_bert, input_attention_mask_bert, batch_lengths,bert_output_ids_masks]]

            #Use current model to predict
            pred_logits, pred_logits_label = model(ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,batch_lengths,max_length) #out:(batch_size,max_length,max_length)
            sentences.extend(batch.sentences)
            #Arc predictions transformed into class-id
            pred = (torch.sigmoid(pred_logits)> 0.5).int()
            pred_matrices_arc.append(pred)
            #Label predictions transformed into class-id
            softmax_probs = torch.softmax(pred_logits_label, dim=-1)  
            pred_label = torch.argmax(softmax_probs, dim=-1)
            pred_matrices_label.append(pred_label)

            pad_mask = mask_to_ignore_pad(max_length+1,batch_lengths,pred_logits)  # (batch_size, max_length, max_length)

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
        print("f1-label",f1_current_label)

        pred_matrices_arc = torch.cat(pred_matrices_arc, dim=0)
        pred_matrices_label = torch.cat(pred_matrices_label, dim=0)

        return(pred_matrices_arc, pred_matrices_label, sentences)

#Fucntion to output graph and save it to a conllu file
def output_graph(out_stream,sentences, pred_matrices_label, pred_matrices_arc):
    batch_output = output_batch_graphs(sentences, pred_matrices_label, pred_matrices_arc)
    
    for sentence_output in batch_output:
        for line in sentence_output:
            out_stream.write(line + '\n')
        out_stream.write('\n')

    out_stream.close()


parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model_save_path', default=None, help='model_save_path') 
parser.add_argument('-test1', '--test_file_1', default=None, help='test_file_1') 
parser.add_argument('-test2', '--test_file_2', default=None, help='test_file_2') 


args = parser.parse_args()
model_save_path = args.model_save_path
path_to_corpus_id= args.test_file_1
path_to_corpu_ood = args.test_file_2

max_length = 200

#Preparation of the dictionary for GLOVE embedding
word_to_idx = prepare_GLOVE_dico('word_to_idx_original.json')
#Prepare the in-domain/out-of-domain test-datasets and creat batches
max_length=200
X_id, Y_id,X_ood, Y_ood = prepare_dataset(path_to_corpus_id,path_to_corpu_ood)
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
batches_id = Batch.create_batches(X_id, Y_id, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True,enable_labels=True)
batches_ood = Batch.create_batches(X_ood, Y_ood, max_length, batch_size=8, tokenizer=tokenizer, glove_w2i=word_to_idx, shuffle=True,enable_labels=True)
print(f'Batches number(id): {len(batches_id)}')
print(f'Batches number(ood): {len(batches_ood)}')


##Reload the trained  model
state_dict = torch.load(model_save_path)
max_length=200
batch_size = 8
embedding_dim=868
hidden_dim_LSTM=600
vocab_size=len(word_to_idx)
output_size_MLP=600
MLP_hidden_layer_size=600
pretrained_embeddings = state_dict['embeddings.weight']
num_labels =31

model=SDP_scorer_arc_label(embedding_dim, hidden_dim_LSTM, vocab_size, output_size_MLP,  MLP_hidden_layer_size, pretrained_embeddings,num_labels,if_freeze_bert=False).to(device)
model.load_state_dict(torch.load(model_save_path))

#use the model to predict and give the overall evaluation of its performance on test-dataset
print("Model performane on in-domain test corpus:")
pred_matrices_arc_id, pred_matrices_label_id, sentences_id = predict(batches_id,model,max_length)
print("Model performane on out-of-domain test corpus:")
pred_matrices_arc_ood, pred_matrices_label_ood, sentences_ood = predict(batches_ood,model,max_length)

#Graph output: in-domain corpus
output_file = 'graph_output_id.conllu'
out_stream_id = open(output_file, 'w')

output_file = 'graph_output_ood.conllu'
out_stream_ood = open(output_file, 'w')

output_graph(out_stream_id,sentences_id, pred_matrices_label_id, pred_matrices_arc_id)
output_graph(out_stream_ood,sentences_ood, pred_matrices_label_ood, pred_matrices_arc_ood)
