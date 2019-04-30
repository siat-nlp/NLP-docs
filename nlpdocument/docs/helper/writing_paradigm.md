# 文档编写模板

```markdown
## Network architecture
### Seq2Seq
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
```
效果：

## Network architecture
### Seq2Seq
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
