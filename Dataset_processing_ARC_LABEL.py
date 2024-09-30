#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

class Corpus:
    def __init__(self, examples):
        self.examples = examples

    def get_example(self, index):
        return self.examples[index]

    def size(self):
        return len(self.examples)

    def summary(self):
        # Provide a summary of the corpus, like number of examples, average length, etc.
        total_examples = len(self.examples)
        total_lines = sum(len(example) for example in self.examples)
        average_length = total_lines / total_examples
        return {
            "total_examples": total_examples,
            "total_lines": total_lines,
            "average_example_length": average_length
        }

class SDPCorpusReader:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def read_corpus(self):
        examples = []
        current_example = []
        with open(self.corpus_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    if current_example:
                        examples.append(current_example)
                    current_example = []
                else:
                    current_example.append(line)
        if current_example:
            examples.append(current_example)
        return Corpus(examples)


class DataProcessor:
    def __init__(self, enable_labels):
        self.enable_labels = enable_labels

    def get_X_y_train(self, examples_list, labels_to_index=None):

        X_train = []
        y_train = []

        for example in examples_list:
            words_list = []
            predicates_index = []
            root_index = None
            for line in example:
                columns = line.strip().split()
                word_index = int(columns[0])
                word = columns[1]
                root = columns[4]
                predicate = columns[5]
                words_list.append(word)
                if predicate == '+':
                    predicates_index.append(word_index)
                if root == '+':
                    root_index = word_index

            if self.enable_labels:
                y_matrix = self.get_labels_dependency_matrix(example, predicates_index, root_index, labels_to_index)
            else:
                y_matrix = self.get_dependency_matrix(example, predicates_index, root_index)

            X_train.append(words_list)
            y_train.append(y_matrix)

        return X_train, y_train

    # This function will be used when enable_labels is True
    def get_labels_dependency_matrix(self, example, predicates_index, root_index, labels_to_index=None):
        n = len(example)
        matrix = torch.zeros((n, n))

        if labels_to_index is None:
            labels_to_index = self.get_labels_to_index()

        for i in range(len(predicates_index)):
            for line in example:
                columns = line.strip().split()
                predicate_index = predicates_index[i] - 1
                word_index = int(columns[0]) - 1
                dependency_label = columns[7 + i]
                if 'comp_' in dependency_label:
                    dependency_label = 'comp'
                if dependency_label != "_":
                    label_index = labels_to_index.get(dependency_label, labels_to_index['unspecified'])
                    matrix[predicate_index][word_index] = label_index

        extended_matrix = torch.zeros((n + 1, n + 1))   # add root row/column
        extended_matrix[1:, 1:] = matrix
        if root_index is not None:
          extended_matrix[0,root_index] = labels_to_index.get('root', labels_to_index['unspecified'])

        return extended_matrix

    # This function will be used when enable_labels is False
    def get_dependency_matrix(self, example, predicates_index, root_index):
        n = len(example)
        matrix = torch.zeros((n, n))

        for i in range(len(predicates_index)):
            for line in example:
                columns = line.strip().split()
                predicate_index = predicates_index[i] - 1
                word_index = int(columns[0]) - 1
                if columns[7 + i] != "_":
                    matrix[predicate_index][word_index] = 1
                    
        extended_matrix = torch.zeros((n + 1, n + 1)) # add root row/column
        extended_matrix[1:, 1:] = matrix
        if root_index is not None:
            extended_matrix[0,root_index] = 1

        return extended_matrix

    @staticmethod
    def get_labels_to_index():
      return {
          'compound': 1, 'ARG1': 2, 'measure': 3, 'ARG2': 4, 'BV': 5,
          'of': 6, 'loc': 7, 'appos': 8, 'ARG3': 9, 'mwe': 10,
          'poss': 11, '_and_c': 12, 'times': 13, 'than': 14, 'part': 15,
          'subord': 16, 'conj': 17, 'comp': 18, 'neg': 19, '_or_c': 20,
          '_but_c': 21, 'plus': 22, 'ARG4': 23, '_as+well+as_c': 24, 'temp': 25,
          'discourse': 26, 'parenthetical': 27, 'manner': 28, 'unspecified': 29, 'root': 30
          }
      


class Batch:
    def __init__(self, glove_ids, bert_input_ids, bert_input_attention_mask, bert_output_ids_mask, y_label, lengths, sentences, enable_labels=False):
        self.glove_ids = glove_ids
        self.bert_input_ids = bert_input_ids
        self.bert_input_attention_mask = bert_input_attention_mask
        self.bert_output_ids_mask = bert_output_ids_mask
        self.y_label = y_label
        self.lengths = lengths
        self.sentences = sentences
        if enable_labels:
            self.y_arc = (y_label != 0).float()

    def __iter__(self):
        elements = (self.glove_ids, self.bert_input_ids, self.bert_input_attention_mask, self.bert_output_ids_mask, self.y_label, self.lengths)
        if hasattr(self, 'y_arc'):
            elements += (self.y_arc,)
        return iter(elements)

    @staticmethod
    def create_batches(X_train, y_train, max_length, batch_size, tokenizer, glove_w2i, shuffle=True, enable_labels=False):
        data_size = len(X_train)
        batches = []
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            batch_indices = indices[start_idx:end_idx]

            sentences = [X_train[i] for i in batch_indices]
            y_label_matrices = [y_train[i] for i in batch_indices]
            y_label = []

            for y_label_matrix in y_label_matrices:
                padded_matrix = Batch.resize_matrix(y_label_matrix, max_length + 1)
                y_label.append(padded_matrix)
            y_label = torch.stack(y_label).float()

            lengths = torch.tensor([min(len(sentence), max_length) + 1 for sentence in sentences], dtype=torch.long)
            glove_ids = Batch.get_glove_ids(sentences, glove_w2i, max_length)

            tokenized_sentences = tokenizer(
                sentences,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=max_length + 2,
                return_tensors="pt"
            )

            bert_input_ids = tokenized_sentences['input_ids']
            bert_input_attention_mask = tokenized_sentences['attention_mask']
            bert_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in bert_input_ids]

            bert_output_ids_mask = torch.stack([Batch.get_mask(tokens_list, max_length) for tokens_list in bert_tokens])

            batch = Batch(glove_ids, bert_input_ids, bert_input_attention_mask, bert_output_ids_mask, y_label, lengths, sentences, enable_labels)

            batches.append(batch)

        return batches

    @staticmethod
    def resize_matrix(matrix, new_size):
        current_size = matrix.size(0)
        if current_size == new_size:
            return matrix
        elif new_size > current_size:
            padded_matrix = torch.zeros((new_size, new_size), dtype=matrix.dtype, device=matrix.device)
            padded_matrix[:current_size, :current_size] = matrix
            return padded_matrix
        else:
            return matrix[:new_size, :new_size]

    @staticmethod
    def get_mask(bert_tokens_list, max_length):
        mask = [not (token.startswith("##") or token in ['[SEP]', '[CLS]']) for token in bert_tokens_list]
        while len(mask) < max_length:
            mask.append(False)
        return torch.tensor(mask)

    @staticmethod
    def get_glove_ids(sentences, glove_w2i, max_length):
        glove_ids = []
        for sentence in sentences:
            lower_sentence = [word.lower() for word in sentence]
            padded_sentence = (lower_sentence + ['[PAD]'] * max_length)[:max_length]
            sentence_glove_ids = [glove_w2i.get(word, 400001) for word in padded_sentence]  # Assuming 400001 is the index for unknown words
            glove_ids.append(sentence_glove_ids)
        return torch.tensor(glove_ids)

    def __repr__(self):
        representation = f"Batch(sentences={self.sentences}\nglove_ids={self.glove_ids.shape}\nbert_ids={self.bert_input_ids.shape}\nattention_mask={self.bert_input_attention_mask.shape}\nbert_output_ids_mask={self.bert_output_ids_mask.shape}\ny_label={self.y_label.shape}\nlengths={self.lengths})"
        if hasattr(self, 'y_arc'):
            representation += f"\ny_arc={self.y_arc.shape}"
        return representation




