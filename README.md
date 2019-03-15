# NLP-DL-Tricks
This repository aims to keep track of some practical and theoretical tricks in natural language processing (NLP) / deep learning / machine learning, etc. Most of these tricks are summarized by members of our group, while some others are borrowed from open-source sites.

#### Table of Contents

* [Data prepossessing](#data-prepossessing)
* [Network architecture](#network-architecture)
* [Parameters](#parameters)
* [Regularization](#regularization)
* [Programming in Tensorflow](#programming-in-tensorflow)
* [Programming in PyTorch](#programming-in-pytorch)
* [Reinforcement learning](#reinforcement-learning)


## Data prepossessing
* **1**：


## Network architecture
* **1. Seq2Seq**: Some tricks to train RNN and seq2seq models:
   * Embedding size: 1024 or 512. Lower dimensionality like 256 can also lead to good performances. Higher does not necessarily lead to better performances.
   * For the decoder: LSTM > GRU > Vanilla-RNN
   * 2-4 layers seems generally enough. Deeper models with residual connections seems more difficult to converge (high variance). More tricks needs to be discovered.
   * ResD (dense residual connections) > Res (only connected to previous layer) > no residual connections
   * For encoder: Bidirectional > Unidirectional (reversed input) > Unidirectional
   * Attention (additive) > Attention (multiplicative) > No attention. Authors suggest that attention act more as a skip connection mechanism than as a memory for the decoder.

  **Ref**: [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906), Denny Britz, Anna Goldie et al.

* **2. Seq2Seq**: For seq2seq, reverse the order of the input sequence (\['I', 'am', 'hungry'\] becomes \['hungry', 'am', 'I'\]). Keep the target sequence intact.

  **Why**: From the authors: "*This way, [...] that makes it easy for SGD to “establish communication” between the input and the output. We found this simple data transformation to greatly improve the performance of the LSTM.*"
  
  **Ref**: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), Ilya Sutskever et al.

* **3. Char-RNN**: By training in an unsupervised way a network to predict the next character of a text (char-RNN), the network will learn a representation which can then be used for a supervised task (here sentiment analysis).

  **Ref**: [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444), Ilya Sutskever et al.



## Parameters
* **1. Learning rate**: The learning rate can be usually initialized as 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1(3x growing up). A strategy used to select the hyperparameters is to randomly sample them (uniformly or logscale) and see the testing error after a few epoch.

* **2. Beam size**: Usually set from 2 to 10. The larger beam size, the higher computational cost.


## Regularization
* **1. Dropout**: To make Dropout works with RNN, it should only be applied on non-recurrent connections (between layers among a same timestep) [1]. Some more recent paper propose some tricks to make dropout works for recurrent connections [2]. 

  **Ref**:
    [1]. [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329), Wojciech Zaremba et al.</br>
    [2]. [Recurrent Dropout without Memory Loss](https://arxiv.org/abs/1603.05118), Stanislau Semeniuta et al.

* **2. Batch normalization**: adding a new normalization layer. Some additional tricks for accelerating BN Networks:
   * Increase the learning rate
   * Remove/reduce Dropout: speeds up training, without increasing overfitting
   * Remove/Reduce the L2 weight regularization
   * Accelerate the learning rate decay: because the network trains faster
   * Remove Local Response Normalization
   * Shuffle training examples more thoroughly: prevents the same examples from always appearing in a mini-batch together. (The authors speak about 1% improvements in the validation)
   * Reduce the photometric distortions

  **Why**: Some good explanation at [Quora](https://www.quora.com/Why-does-batch-normalization-help).


## Programming in Tensorflow
* **1. ```tf.variable_scope```/```tf.name_scope```**: Both scopes have the same effect on all operations as well as variables, but *name scope* is ignored by ```tf.get_variable```. Suggest use ```tf.variable_scope``` in most cases. 

  **Ref**: The difference between name scope and variable scope in tensorflow at [stackoverflow](https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow).

* **2. Model Save/Restore**: Usually, we create a helper ```saver = tf.train.Saver()``` to save and restore the whole model. However, if we want to use pre-trained model for fine-tuning or transfer learning, there are 2 ways: (1) Create the network by writing code to create each and every layer manually as the original model, and then use ```tf.train.Saver()``` to restore pre-trained model's checkpoint file. (2) Use ```.meta``` file and create the helper as ```saver = tf.train.import_meta_graph('xxx_model-xxx.meta')``` and then restore the pre-trained model. 

  **Ref**: More details are in this [tutorial](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/).


## Programming in PyTorch
* **1. ```CUDA out of memory```**: When ```RuntimeError: CUDA out of memory``` occurs, usually (1) check if exists too large tensors in computation graph; (2) downsize the batch size; (3) or use multiple GPUs to train. Note to split batch size when using ```nn.DataParallel```. 

  **Ref**: Some other details are in this [debug log](https://docs.google.com/document/d/1Cpxs-aZcydqCzTEvfW-62ja6ZDhx2QEXR-f5HKmbeig/edit?usp=sharing).


## Reinforcement learning
* **1. Asynchronous**: Train simultaneously multiple agents with different exploration policies (e.g., E-greedy with different values of epsilon) improve the robustness. 

  **Ref**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), V. Mnih.

* **2. Skip frame**: Compute the action every 4 frames instead of every frames. For the other frames, repeat the action. 

  **Why**: Works well on Atari games, when the player reactivity doesn't need to be frame perfect. Using this trick allows to greatly speed up the training (About x4). 

  **Ref**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), V. Mnih.

* **3. History**: Instead of only taking the current frame as input, stack the last frames together on a single input (size (h, w, c) with 1 grayscale frame by channel). Combined with a skip frame (repeat action) of 4, that means we would stack the frames t, t-4, t-8 and t-12. 

  **Why**: This allows the network to have some momentum information. 

  **Ref**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), V. Mnih.

* **4. Experience Replay**: Instead of updating every frames as the agent plays, to avoid correlations between the frames, it's better to sample a batch in the history of the transition taken (state, actionTaken, reward, nextState). This is basically the same idea as shuffling the dataset before training for supervised tasks. Some strategies exist to sample batches which contain more information (in the sense predicted reward different from real reward). 

  **Ref**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Tom Schaul et al.

* **5. PAAC (Parallel Advantage Actor Critic)**: It's possible to simplify the the A3C algorithm by batching the agent experiences and using a single model with synchronous updates. 

  **Ref**: [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862v2), Alfredo V. Clemente et al.

