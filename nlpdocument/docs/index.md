# NLP and Deep Learning Tricks
This repository aims to keep track of some practical and theoretical tricks in natural language processing (NLP) / deep learning / machine learning, etc. Most of these tricks are summarized by members of our group, while some others are borrowed from open-source sites.



## Data prepossessing

## Network architecture
### **Seq2Seq**
Some tricks to train RNN and seq2seq models:

* Embedding size: 1024 or 512. Lower dimensionality like 256 can also lead to good performances. Higher does not necessarily lead to better performances.
* For the decoder: LSTM > GRU > Vanilla-RNN
* 2-4 layers seems generally enough. Deeper models with residual connections seems more difficult to converge (high variance). More tricks needs to be discovered.
* ResD (dense residual connections) > Res (only connected to previous layer) > no residual connections
* For encoder: Bidirectional > Unidirectional (reversed input) > Unidirectional
* Attention (additive) > Attention (multiplicative) > No attention. Authors suggest that attention act more as a skip connection mechanism than as a memory for the decoder.

!!! info "Ref"
    [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906), Denny Britz, Anna Goldie et al.

For seq2seq, reverse the order of the input sequence (\['I', 'am', 'hungry'\] becomes \['hungry', 'am', 'I'\]). Keep the target sequence intact.

!!! question "Why"
    From the authors: "*This way, [...] that makes it easy for SGD to “establish communication” between the input and the output. We found this simple data transformation to greatly improve the performance of the LSTM.*"

!!! info "Ref"
    [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), Ilya Sutskever et al.

<br />
### **Char-RNN** 
By training in an unsupervised way a network to predict the next character of a text (char-RNN), the network will learn a representation which can then be used for a supervised task (here sentiment analysis).

!!! info "Ref"
    [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444), Ilya Sutskever et al.


## Parameters
### Learning rate
The learning rate can be usually initialized as 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1(3x growing up). A strategy used to select the hyperparameters is to randomly sample them (uniformly or logscale) and see the testing error after a few epoch.

### Beam size
Usually set from 2 to 10. The larger beam size, the higher computational cost.

## Regularization
### Dropout
To make Dropout works with RNN, it should only be applied on non-recurrent connections (between layers among a same timestep) [1]. Some more recent paper propose some tricks to make dropout works for recurrent connections[2].

!!! info "Ref"
    [1]. [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329), Wojciech Zaremba et al.</br>
    [2]. [Recurrent Dropout without Memory Loss](https://arxiv.org/abs/1603.05118), Stanislau Semeniuta et al.

### Batch normalization 
adding a new normalization layer. Some additional tricks for accelerating BN Networks:
   * Increase the learning rate
   * Remove/reduce Dropout: speeds up training, without increasing overfitting
   * Remove/Reduce the L2 weight regularization
   * Accelerate the learning rate decay: because the network trains faster
   * Remove Local Response Normalization
   * Shuffle training examples more thoroughly: prevents the same examples from always appearing in a mini-batch together. (The authors speak about 1% improvements in the validation)
   * Reduce the photometric distortions

!!! question "Why"
    Some good explanation at [Quora](https://www.quora.com/Why-does-batch-normalization-help).



## Reinforcement learning
### Asynchronous
Train simultaneously multiple agents with different exploration policies (e.g., E-greedy with different values of epsilon) improve the robustness. 

!!! info "Ref"
    [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), V. Mnih.

### Skip frame
Compute the action every 4 frames instead of every frames. For the other frames, repeat the action. 

!!! question "Why"
    Works well on Atari games, when the player reactivity doesn't need to be frame perfect. Using this trick allows to greatly speed up the training (About x4). 

!!! info "Ref"
    [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), V. Mnih.

### History
Instead of only taking the current frame as input, stack the last frames together on a single input (size (h, w, c) with 1 grayscale frame by channel). Combined with a skip frame (repeat action) of 4, that means we would stack the frames t, t-4, t-8 and t-12. 

!!! question "Why"
    This allows the network to have some momentum information. 

!!! info "Ref"
    [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), V. Mnih.

### Experience Replay
Instead of updating every frames as the agent plays, to avoid correlations between the frames, it's better to sample a batch in the history of the transition taken (state, actionTaken, reward, nextState). This is basically the same idea as shuffling the dataset before training for supervised tasks. Some strategies exist to sample batches which contain more information (in the sense predicted reward different from real reward). 

!!! info "Ref"
    [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), Tom Schaul et al.

### PAAC (Parallel Advantage Actor Critic)
It's possible to simplify the the A3C algorithm by batching the agent experiences and using a single model with synchronous updates. 

!!! info "Ref"
    [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862v2), Alfredo V. Clemente et al.
