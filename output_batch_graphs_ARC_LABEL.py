import torch

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



def output_batch_graphs(sentences, y_label, y_arc, index_to_label=None):
    if y_label.shape != y_arc.shape:
        raise ValueError("y_label and y_arc must have the same dimensions.")

    # Default to index_to_label dictionary below
    if index_to_label is None:
        index_to_label = {
            1: 'compound', 2: 'ARG1', 3: 'measure', 4: 'ARG2', 5: 'BV',
            6: 'of', 7: 'loc', 8: 'appos', 9: 'ARG3', 10: 'mwe',
            11: 'poss', 12: '_and_c', 13: 'times', 14: 'than', 15: 'part',
            16: 'subord', 17: 'conj', 18: 'comp', 19: 'neg', 20: '_or_c',
            21: '_but_c', 22: 'plus', 23: 'ARG4', 24: '_as+well+as_c', 25: 'temp',
            26: 'discourse', 27: 'parenthetical', 28: 'manner', 29: 'unspecified', 30: 'root'
        }

    batch_outputs = []      
    
    for batch_idx, sentence in enumerate(sentences):
        dependency_matrix = y_label[batch_idx] * y_arc[batch_idx]
        n = len(sentence)
        outputs = [f"# Example {batch_idx + 1}"]
        dependency_matrix = resize_matrix(dependency_matrix,n+1) # n+1 to keep root

        for i in range(n):
            word = sentence[i]
            heads = []
            dependency_type = []
            heads_column = dependency_matrix[:, i+1]  # Get the i+1 column for the current word
            for index, number in enumerate(heads_column):
                if number != 0:
                    heads.append(str(index))
                    number = number.item()
                    dependency_type.append(index_to_label[number])

            head_str = "|".join(heads)
            dependency_type_str = "|".join(dependency_type)
            
            # Prepare the output string
            max_word_length = max(len(word) for word in sentence) + 1
            outputs.append(f"{i+1:<3} {word:<{max_word_length}} {head_str:<5} {dependency_type_str}")

        batch_outputs.append(outputs)

    return batch_outputs
