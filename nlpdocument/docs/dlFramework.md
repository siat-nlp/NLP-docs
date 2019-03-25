# Deep Learning Framework Programming


## Programming in Tensorflow
### tf.variable_scope/tf.name_scope
Both scopes have the same effect on all operations as well as variables, but *name scope* is ignored by ```:::python tf.get_variable```. Suggest use ```:::python tf.variable_scope``` in most cases. 

!!! info "Ref"
    The difference between name scope and variable scope in tensorflow at [stackoverflow](https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow).

### Model Save/Restore
Usually, we create a helper ```saver = tf.train.Saver()``` to save and restore the whole model. However, if we want to use pre-trained model for fine-tuning or transfer learning, there are 2 ways: (1) Create the network by writing code to create each and every layer manually as the original model, and then use ```tf.train.Saver()``` to restore pre-trained model's checkpoint file. (2) Use ```.meta``` file and create the helper as ```saver = tf.train.import_meta_graph('xxx_model-xxx.meta')``` and then restore the pre-trained model. 

!!! info "Ref"
    More details are in this [tutorial](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/).


## Programming in PyTorch
### CUDA out of memory
When ```RuntimeError: CUDA out of memory``` occurs, usually (1) check if exists too large tensors in computation graph; (2) downsize the batch size; (3) or use multiple GPUs to train. Note to split batch size when using ```nn.DataParallel```. 

!!! info "Ref"
    Some other details are in this [debug log](https://docs.google.com/document/d/1Cpxs-aZcydqCzTEvfW-62ja6ZDhx2QEXR-f5HKmbeig/edit?usp=sharing).