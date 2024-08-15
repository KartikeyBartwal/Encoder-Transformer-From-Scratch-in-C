 ![BERT arch 2](https://github.com/user-attachments/assets/6b890804-f0ff-44c2-adec-872dc7d8f922)
 
# BERT from Scratch in C

This project is an ambitious attempt to build the entire BERT (Bidirectional Encoder Representations from Transformers) architecture using the C programming language. The following sections outline the workflow and components of the implementation.

## Project Overview

1. **Text Data Preparation**:
   - The input text data is sourced from `text_data.txt`.
   - Basic text preprocessing functions are applied to clean the data, including:
     - Converting text to lowercase
     - Removing complex punctuation

2. **Sentence Splitting**:
   - The cleaned corpus is split into individual sentences.
   - A numeric mapping is created for each unique word present in the text.

3. **Token Embedding**:
   - The token embedding process involves adding special tokens:
     - For example, given the sentences: “The cat is walking. The dog is barking,” the input becomes:
       ```
       [CLS] the cat is walking [SEP] the dog is barking
       ```

4. **Segment Embedding**:
   - Segment embeddings are created to differentiate between the two sentences:
     ```
     [0, 0, 0, 0, 1, 1, 1, 1]
     ```
     
![Screenshot from 2024-08-15 12-32-22](https://github.com/user-attachments/assets/c6d9b902-50fe-4d01-9ddf-756450fc5168)

5. **Masking**:
   - Random masks are assigned to 15% of the sequence according to the original BERT paper:
     - 80% of the masked tokens are replaced with `[MASK]`
     - 10% are replaced with random tokens
     - 10% remain unchanged

6. **Padding**:
   - After masking, padding is applied to ensure uniform sequence lengths up to a specified maximum length (`max_len`).

## BERT Architecture Components

### A. Embedding Layer
The embedding layer generates:
1. **Token Embeddings**: Represents the input tokens.
2. **Position Embeddings**: Encodes the position of each token in the sequence.
3. **Segment Embeddings**: Indicates the sentence from which the token originates.
   
![BERT-input-representation-Sum-of-segment-position-and-token-embeddings-is-the-input](https://github.com/user-attachments/assets/99ca6326-9b11-46f0-b80c-76f219624c01)

### B. Self-Attention Layer
- The resulting vector from the embedding layer passes through a self-attention mechanism that captures dependencies between tokens in the sequence.

![Screenshot from 2024-08-15 12-39-09](https://github.com/user-attachments/assets/b582f556-845a-45b0-802b-5b38738dffdb)

### C. Feedforward Layers
1. The output from the self-attention layer is passed through a feedforward layer followed by a Tanh activation function.
2. It then goes through another feedforward layer to reduce the dimensionality back to the original embedding size.

### D. Additional Layers
1. **Masked Token Prediction**: Enables the model to predict missing words in a sentence or perform next-word prediction.
2. **Next Sentence Prediction**: Extends the capabilities of the model to understand sentence relationships.

### E. Backpropagation
- The project incorporates backpropagation, which is essential for training deep learning models. Each layer computes its own loss function, facilitating the update of weights and parameters in the self-attention layer, feedforward layers, and the three embedding layers.
