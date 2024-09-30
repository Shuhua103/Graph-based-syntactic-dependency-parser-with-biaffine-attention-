import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# 创建一个函数来移除全 0 的嵌入，并补全到原始长度
def remove_and_pad(embeddings, mask, max_length):
    new_embeddings = []

    for i in range(embeddings.size(0)):
        valid_embeds = embeddings[i][mask[i].any(dim=-1)]  # 提取有效嵌入
        num_valid = valid_embeds.size(0)
        if num_valid > max_length:
            valid_embeds = valid_embeds[:max_length]  # 截断超过 max_length 的部分
            num_valid = max_length
        num_padding = max_length - num_valid
        padding = torch.zeros((num_padding, embeddings.size(2)), device=embeddings.device)
        new_embed = torch.cat((valid_embeds, padding), dim=0)
        new_embeddings.append(new_embed)

    return torch.stack(new_embeddings)


from transformers import AutoModel, AutoConfig
class SDP_scorer_arc_label(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size,  MLP_hidden_layer_size, pretrained_embeddings,num_labels,if_freeze_bert=True):
        """ -embedding_dim：the size of the word-embedding
            -hidden_dim：the size of the recurrent representation gotten from biLSTM
            -MLP_hidden_layer_size: the size of the hidden layer of the MLPs
            -output_size：the size of the out put from the MLPs
            -embeddings: if set as none, then initialize a new naive embedding matrix
            -vocab_size: size of voc for initializing the embedding matrix
        """
        super(SDP_scorer_arc_label, self).__init__()

        #Register Glove Embeddings
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)

        #register non-freezon bert to be trainable
        self.bert_layer = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_config = AutoConfig.from_pretrained('distilbert-base-uncased')

        # Free ze bert model parameters or not
        if if_freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
            print("Bert is frozen")

        #biLSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=3)
        #biLSTM layer's dropout layer
        self.dropout_layer_biLSTM = nn.Dropout(p=0.33)

        #MLP(head)
        self.MLP_head = nn.Sequential(nn.Linear(2*hidden_dim, MLP_hidden_layer_size),
                                      nn.ReLU(),nn.Dropout(p=0.33),nn.Linear(MLP_hidden_layer_size, output_size))

        #MLP(dep)
        self.MLP_dep = nn.Sequential(nn.Linear(2*hidden_dim, MLP_hidden_layer_size),
                                      nn.ReLU(),nn.Dropout(p=0.33),nn.Linear(MLP_hidden_layer_size, output_size))
        
        #MLP(label's head)
        self.MLP_label_head = nn.Sequential(nn.Linear(2*hidden_dim, MLP_hidden_layer_size),
                                      nn.ReLU(),nn.Dropout(p=0.33),nn.Linear(MLP_hidden_layer_size, output_size))

        #MLP(label's dep)
        self.MLP_label_dep = nn.Sequential(nn.Linear(2*hidden_dim, MLP_hidden_layer_size),
                                      nn.ReLU(),nn.Dropout(p=0.33),nn.Linear(MLP_hidden_layer_size, output_size))


        #Biaffine mechanism(simplified)
        #a weight matrix of shape (output_size of MLP, output_size of MLP)
        self.U_biaffine = nn.Linear(output_size, output_size, bias=False)
        #a bias(a real number) initialized as zero
        self.b_biaffine = nn.Parameter(torch.zeros(1))

        #Biaffine matrix for arc existence (output_size of MLP, output_size of MLP)
        self.U_biaffine = nn.Linear(output_size, output_size, bias=False)
        #a bias(a real number) initialized as zero
        self.b_biaffine = nn.Parameter(torch.zeros(1))

        #Biaffine matrix for labels (num_labels,output_size of MLP, output_size of MLP)
        self.U_biaffine_label = nn.Parameter(torch.randn(num_labels, output_size, output_size))
        self.b_biaffine_label = nn.Parameter(torch.zeros(num_labels)) 

        #a special embed for the fake token [ROOT]
        self.root_embed = nn.Parameter(torch.randn(1, 1, embedding_dim))  # (1, 1, embedding_dim)


    def forward(self, ids_glove_batch,input_ids_bert,input_attention_mask_bert,bert_output_ids_masks,batch_lengths,max_length):
        """ -batch：input tensor: shape (batch_size, max_length)
            -batch_lengths：a tensor registering sequence lengths without padding: shape (batch_size)
        """

        #Look up the GLOVE embeddings
        embeddings_glove = self.embeddings(ids_glove_batch) #out shape:（batch_size, max_length, embedding_size_glove）

        # Get BERT embeddings
        outputs = self.bert_layer(input_ids=input_ids_bert, attention_mask=input_attention_mask_bert)
        embeddings_bert = outputs.last_hidden_state #out shape:（batch_size, padded_sequence_length, embedding_size_bert）

        # Use bert_output_ids_masks to keep valid subtokens embeddings:
        # 将掩码矩阵扩展到与 BERT 嵌入矩阵相同的形状
        bert_output_masks = bert_output_ids_masks.unsqueeze(-1).expand_as(embeddings_bert)

        # 使用掩码矩阵将不需要的嵌入设置为零
        masked_embeddings = embeddings_bert * bert_output_masks
        # 应用函数
        embeddings_bert = remove_and_pad(masked_embeddings, masked_embeddings,max_length)

        #concatenation of bert-embedding and GLOVE embedding
        embeddings = torch.cat((embeddings_bert, embeddings_glove), dim=2)#out shape:（batch_size, max_length, embedding_size_bert+embedding_size_glove）

        # x 是输入数据，假设形状为 (batch_size, seq_len, embedding_dim)
        # 扩展 ROOT 嵌入以匹配批次大小
        root_embeds = self.root_embed.expand(embeddings.size(0), -1, -1)  # (batch_size, 1, embedding_dim)    
        
        # 将 ROOT 嵌入添加到每个序列的前面
        embeddings = torch.cat([root_embeds, embeddings], dim=1)  # (batch_size, seq_len + 1, embedding_dim)


        #BiLSTM : get contextualized reccurent representation for every token
        batch_lengths = batch_lengths.to('cpu').long()
        #sequence packing, padded parts temporarily ignored
        packed_seqs = pack_padded_sequence(embeddings, batch_lengths, batch_first=True, enforce_sorted=False)#out：an instance of PackedSequence
        #pass through the biLSTM layer
        lstm_output, _ = self.lstm(packed_seqs) #out：an instance of PackedSequence
        
        #unpacking, the original shape restored
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True, total_length=max_length+1)
        # lstm_output, _ = pad_packed_sequence(lstm_output) #out:(max_length, batch_size, 2*hidden_dim)
        #pass the dropout layer
        lstm_output = self.dropout_layer_biLSTM(lstm_output) #(batch_size, max_length, 2*hidden_dim)



        #MLPs: characterize the token representation with head/dep features
        input_MLPs = lstm_output#(batch_size, max_length,  2*hidden_dim)

        #FORWARD-ARC
        #pass MLP-head
        H_head = self.MLP_head(input_MLPs)#out (batch_size, max_length, output_size)

        #pass MLP-dep
        H_dep = self.MLP_dep(input_MLPs)#out (batch_size, max_length, output_size)

        #Biaffine transformation: get the score(logit) for every pair (head,dep)
        #H(head) · U(arc)
        Head_U = self.U_biaffine(H_head) #out:(batch_size,max_length,output_size)

        #H(head) · U(arc) · H(dep)
        Head_U_dep = Head_U.matmul(H_dep.transpose(1,2))#out:(batch_size,max_length,max_length)

        #Plus the bias to every cell(=plus the bias to every score(head,dep) gotten from biaffine linear combination)
        logits= Head_U_dep + self.b_biaffine #out:(batch_size,max_length,max_length)


        ## FORWARD-LABEL
        #pass MLP-head
        L_head = self.MLP_label_head(input_MLPs)#out (batch_size, max_length, output_size)

        #pass MLP-dep
        L_dep = self.MLP_label_dep(input_MLPs)#out (batch_size, max_length, output_size)

        # H(head) · U(label)
        Head_U = torch.einsum('bxi,oij->bxoj', L_head, self.U_biaffine_label)  # out: (batch_size, max_length, num_labels, mlp_dim)

        # H(head) · U(arc) · H(dep)
        Head_U_dep = torch.einsum('bxoj,byj->bxyo', Head_U, L_dep)  # (batch_size, max_length, max_length, num_labels)
        # Add the bias vector
        logits_label = Head_U_dep + self.b_biaffine_label.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (batch_size, max_length, max_length, num_labels)
        
        # print(f"logits ", {logits})
        return(logits,logits_label)
    


