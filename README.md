### Building a Transformer Model from Scratch in C

In this project, I implemented a complete Transformer model from scratch using the C programming language. The model mimics the architecture described in the original "Attention is All You Need" paper, featuring components such as self-attention, positional encoding, feed-forward networks, and multi-layer perceptrons. Below is a breakdown of the project's file structure, detailing each module and its role in the overall architecture.

### File Structure Overview

```plaintext
├── my_terminal_output.log
├── output.txt
├── python.py
├── self_attention_layer.c
├── Some Images
│   ├── feature_engineering.png
│   ├── layer_1_word_embedding.png
│   ├── layer_2_positional_encoding.png
│   ├── layer_3_self-attention-block.png
│   ├── output_generation.png
│   ├── training_data_generation.png
│   └── word_embedding_updated.png
├── test.c
├── text_data.txt
├── Tokenizer.c
├── Tokenizer.h
├── training_data_book.txt
├── training_data_book_.txt
├── transformer_block.c
├── transformer_block.h
└── Transformer Model 
    ├── final_multi-layer_perceptron.c
    ├── positional_encoding.c
    ├── self_attention.c
    └── word_embedding.c

├── activation_functions.c
├── activation_functions.h
├── a.out
├── backpropagation.c
├── backpropagation.h
├── best_so_far_output.txt
├── Data_Loading_Cleaning.c
├── Data_Loading_Cleaning.h
├── Data_Preprocessing.c
├── Data_Preprocessing.h
├── feed_forward_layer.c
├── feed_forward_layer.h
├── main.c
├── Model Trained Weights
│   ├── create_files.sh
│   ├── Final Multi-layered perceptron weights
```

### Detailed Explanation of Each Component

1. **Core Components of the Transformer**
   - **`transformer_block.c` and `transformer_block.h`**: These files contain the main Transformer block logic, which integrates all core modules (self-attention, feed-forward networks, etc.). This block is used to stack multiple layers, replicating the Transformer architecture's encoder and decoder layers.
   
   - **`self_attention_layer.c`**: Implements the self-attention mechanism that allows the model to focus on different parts of the input sequence dynamically. This module computes the attention scores and applies them to the input data.

   - **`positional_encoding.c`**: Handles the positional encoding, which adds information about the position of each word in the input sequence, allowing the model to understand the order of words.

   - **`feed_forward_layer.c` and `feed_forward_layer.h`**: Define the feed-forward network used in each Transformer layer after the self-attention mechanism. This network adds non-linearity and helps the model to learn complex patterns.

   - **`final_multi-layer_perceptron.c`**: Contains the implementation of the final Multi-Layer Perceptron (MLP) that converts the processed data into the desired output format, such as generating the next word in a sequence.

   - **`word_embedding.c`**: Implements the word embedding layer that converts input words into continuous vector representations, enabling the model to process them effectively.

2. **Training and Data Preparation**
   - **`Data_Loading_Cleaning.c` and `Data_Loading_Cleaning.h`**: These files are responsible for loading and cleaning the raw data. This includes removing unwanted characters, handling missing data, and normalizing the text.

   - **`Data_Preprocessing.c` and `Data_Preprocessing.h`**: Handle tokenization and transformation of raw text data into a format suitable for training. This step includes generating word indices and converting them into embedding vectors.

   - **`Tokenizer.c` and `Tokenizer.h`**: Implements the tokenizer to break down the input text into manageable tokens (words or subwords). The tokens are then fed into the model for further processing.

   - **`training_data_book.txt` and `training_data_book_.txt`**: These files contain the raw and preprocessed training data used to train the Transformer model. They provide the sentences and sequences that the model learns from.

   - **`Model Trained Weights`**: This folder stores the trained weights of the model, which are obtained after running the training scripts. The weights can be reused for inference or further fine-tuning.

3. **Optimization and Backpropagation**
   - **`backpropagation.c` and `backpropagation.h`**: Implement the backpropagation algorithm, which is crucial for training the neural network by minimizing the loss function through gradient descent.

   - **`activation_functions.c` and `activation_functions.h`**: Contain various activation functions used throughout the model, such as ReLU, sigmoid, and softmax. These functions introduce non-linearities into the model, allowing it to learn complex patterns.

4. **Execution and Auxiliary Files**
   - **`main.c`**: Serves as the entry point of the program, coordinating the initialization, training, and evaluation of the Transformer model.

   - **`test.c`**: Used for testing individual components and ensuring that each module functions as expected.

   - **`my_terminal_output.log`**: Logs the output generated during the execution of the model, providing insights into the model's performance and training progress.

   - **`output.txt`**: Captures the final output generated by the model, such as predicted sequences or text.

5. **Visualization and Analysis**
   - **`Some Images/`**: Contains various images for visualization purposes, such as:
     - `feature_engineering.png`: Demonstrates the steps involved in feature engineering.
     - `layer_1_word_embedding.png`, `layer_2_positional_encoding.png`, `layer_3_self-attention-block.png`: Visual representations of different layers of the Transformer model.
     - `output_generation.png` and `training_data_generation.png`: Show the process of output generation and training data preparation.

6. **Scripts and Utilities**
   - **`create_files.sh`**: A shell script to automate the creation of necessary files or directories for storing model weights, outputs, and other related data.

   - **`a.out`**: The compiled binary file generated after building the project.

7. **Best Output and Performance**
   - **`best_so_far_output.txt`**: Stores the best output generated during the model's training phase, which can be used as a reference for further improvements.

### Conclusion

This project showcases a complete implementation of the Transformer model from scratch in C, highlighting my understanding of both deep learning concepts and low-level programming. Each module is meticulously crafted to replicate the Transformer architecture, ensuring flexibility and efficiency. By combining different components such as self-attention, positional encoding, feed-forward networks, and optimization algorithms, this implementation demonstrates the power and versatility of Transformers in natural language processing tasks.


![diagram-export-6-9-2024-11_07_47-pm](https://github.com/user-attachments/assets/b443ff78-618f-4ea8-8832-4d703f97bbd9)


![diagram-export-6-9-2024-11_08_50-pm](https://github.com/user-attachments/assets/67e2e762-c7b1-4881-9132-0a1dc9cc2d62)


![diagram-export-6-9-2024-11_20_10-pm](https://github.com/user-attachments/assets/cd8bbdcc-2f11-4ff2-834a-b4c31a946e9d)


![diagram-export-6-9-2024-11_24_50-pm](https://github.com/user-attachments/assets/3c17c0b1-98c8-43c8-888e-72d3d58694cb)
