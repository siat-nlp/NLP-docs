# DL Framework Programming
This repository aims to keep track of some practical and theoretical tricks in natural language processing (NLP) / deep learning / machine learning, etc. Most of these tricks are summarized by members of our group, while some others are borrowed from open-source sites.

## Programming in Tensorflow
### tf.variable_scope/tf.name_scope
Both scopes have the same effect on all operations as well as variables, but *name scope* is ignored by ```tf.get_variable```. Suggest use ```tf.variable_scope``` in most cases. 

> **Ref**: The difference between name scope and variable scope in tensorflow at [stackoverflow](https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow).

### Model Save/Restore
Usually, we create a helper ```saver = tf.train.Saver()``` to save and restore the whole model. However, if we want to use pre-trained model for fine-tuning or transfer learning, there are 2 ways: (1) Create the network by writing code to create each and every layer manually as the original model, and then use ```tf.train.Saver()``` to restore pre-trained model's checkpoint file. (2) Use ```.meta``` file and create the helper as ```saver = tf.train.import_meta_graph('xxx_model-xxx.meta')``` and then restore the pre-trained model. 

> **Ref**: More details are in this [tutorial](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/).


## Programming in PyTorch
