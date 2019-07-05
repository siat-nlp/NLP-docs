# NLP and Deep Learning Tricks
This repository aims to keep track of some practical and theoretical tricks in natural language processing (NLP) / deep learning / machine learning, etc. Most of these tricks are summarized by members of our group, while some others are borrowed from open-source sites.


## Data processing
### Data check
Do remember that carefully checking all the data is the most important preliminary step before building your model. Generally in NLP, we would check following things:
* whether dirty data exixts (e.g., unreadable characters, incomplete key-value pairs) 
* what the max/min/avg lengths of input texts and output texts are
* whether the vocabulary file is correcltly built
* whether the data format at the last step before inputting into the model is actually as you what expect (**very important!**)

### Data chunking
If the data is too large so that it cannot be loaded into the memory at once, we need to chunk the data at this time. Concretely, we split the whole data into certain number of chunk files and store on disk, then maintain a queue for reading chunk files into memory. The dequeue operation takes out certain number of chunk data for building data batches, the enqueue operation reads chunk files into memory one by one from disk to keep the queue full. Here, we provide a class ```DataBatcher``` as a possible implementation:
```
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import time
from queue import Queue
from threading import Thread
from data_loader import DataLoader

# max number of chunk files being loaded into memory
CHUNK_NUM = 20


class DataBatcher(object):
    """
        Data batcher with queue for loading big dataset
    """

    def __init__(self, data_dir, file_list, batch_size, num_epoch, shuffle=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.shuffle = shuffle

        self.cur_epoch = 0
        self.loader_queue = Queue(maxsize=CHUNK_NUM)
        self.loader_queue_size = 0
        self.batch_iter = self.batch_generator()
        self.input_gen = self.loader_generator()

        # Start the threads that load the queues
        self.loader_q_thread = Thread(target=self.fill_loader_queue)
        self.loader_q_thread.setDaemon(True)
        self.loader_q_thread.start()

        # Start a thread that watches the other threads and restarts them if they're dead
        self.watch_thread = Thread(target=self.monitor_threads)
        self.watch_thread.setDaemon(True)
        self.watch_thread.start()

    def get_batch(self):
        try:
            batch_data, local_size = next(self.batch_iter)
        except StopIteration:
            batch_data = None
            local_size = 0
        return batch_data, local_size

    def get_epoch(self):
        return self.cur_epoch

    def full(self):
        if self.loader_queue_size == CHUNK_NUM:
            return True
        else:
            return False

    def batch_generator(self):
        while self.loader_queue_size > 0:
            data_loader = self.loader_queue.get()
            n_batch = data_loader.n_batch
            self.loader_queue_size -= 1
            for batch_idx in range(n_batch):
                batch_data, local_size = data_loader.get_batch(batch_idx=batch_idx)
                yield batch_data, local_size

    def loader_generator(self):
        for epoch in range(self.num_epoch):
            self.cur_epoch = epoch
            if self.shuffle:
                np.random.shuffle(self.file_list)
            for idx, f in enumerate(self.file_list):
                # here, the file reading process may vary from your task
                reader = open("%s/%s" % (self.data_dir, f), 'br')
                chunk_data = pickle.load(reader)
                # here, DataLoader is a self-defined class for data batching, similar to DataLoader in PyTorch
                data_loader = DataLoader(data=chunk_data, batch_size=self.batch_size)
                yield data_loader

    def fill_loader_queue(self):
        while True:
            if self.loader_queue_size <= CHUNK_NUM:
                try:
                    data_loader = next(self.input_gen)
                    self.loader_queue.put(data_loader)
                    self.loader_queue_size += 1
                except StopIteration:
                    break

    def monitor_threads(self):
        """Watch loader queue thread and restart if dead."""
        while True:
            time.sleep(60)
            if not self.loader_q_thread.is_alive():  # if the thread is dead
                print('Found loader queue thread dead. Restarting.')
                new_t = Thread(target=self.fill_loader_queue)
                self.loader_q_thread = new_t
                new_t.daemon = True
                new_t.start()
```


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
