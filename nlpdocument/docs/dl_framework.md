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

## Online Tensorflow Serving
TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. Servables are the central abstraction in TensorFlow Serving. Servables are the underlying objects that clients use to perform computation (for example, a lookup or inference).

The size and granularity of a Servable is flexible. A single Servable might include anything from a single shard of a lookup table to a single model to a tuple of inference models. Servables can be of any type and interface, enabling flexibility and future improvements such as: streaming results, experimental APIs, asynchronous modes of operation.

### How to deploy
Tensorflow Serving follows the server-client architecture. While training a specific model, save it in the mode that can be used by tensorflow-serving. Deploy your model on a running docker to provide service. For clients, request server for prediction results of given data instances. The following is an example of the deployment procedure.

1. Save a trained model
The common way to save model in tensorflow looks like,

        saver.save(session, checkpoint_prefix, global_step=current_step)
        tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
For tensorflow serving, we save like,

        ### define input&output signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'input_x1': tf.saved_model.utils.build_tensor_info(self.input_x1),
            'input_x2': tf.saved_model.utils.build_tensor_info(self.input_x2),
            'ent_x1': tf.saved_model.utils.build_tensor_info(self.ent_x1),
            'ent_x2': tf.saved_model.utils.build_tensor_info(self.ent_x2),
            'input_y': tf.saved_model.utils.build_tensor_info(self.input_y),
            'add_fea': tf.saved_model.utils.build_tensor_info(self.add_fea),
            'dropout_keep_prob': tf.saved_model.utils.build_tensor_info(self.dropout_keep_prob)
        },
        outputs={
            'output': tf.saved_model.utils.build_tensor_info(self.soft_prob)
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
        ### saving 
        builder = tf.saved_model.builder.SavedModelBuilder(graph_save_dir)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()
Define your own inputs and outputs according to the task.

2. Run a serving docker and deploy your model
            
        # pull a tensorflow-serving image
        $ sudo docker pull tensorflow/serving:latest-devel
        # run the serving docker
        $ sudo docker run -it -p 8500:8500 tensorflow/serving:latest-devel
        # copy your model file to the running docker (change the docker ID and your model path)
        $ sudo docker cp /data/huangweiyi/qaModel/code/kaaqa/runs/model 6d7d70e27ecc:/online_qa_model

Note: You need to create different version of your model for tensorflow-serving (refer to https://stackoverflow.com/questions/45544928/tensorflow-serving-no-versions-of-servable-model-found-under-base-path)

            
        ### deploy your model in the running docker
        $ tensorflow_model_server --port=8500 --model_name=qa --model_base_path=/online_qa_model
3. Request the server for prediction results

        hostport = '172.17.0.2:8500'
        channel = grpc.insecure_channel(hostport)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'qa'
    
        inpH = InputHelper()
        x1_test, x2_test, ent_x1_test, ent_x2_test, y_test, x1_temp, x2_temp, add_fea_test = inpH.getTestSample(question, candidates)
        batches = inpH.batch_iter(list(zip(x1_test, x2_test, ent_x1_test, ent_x2_test, y_test, add_fea_test)), 10000, 1, shuffle=False)
        for db in batches:
            x1_dev_b, x2_dev_b, ent_x1_dev_b, ent_x2_dev_b, y_dev_b, add_fea_dev_b = zip(*db)
            for idx in range(len(x1_dev_b)):
                feature_dict = {
                    "input_x1": x1_dev_b,
                    "input_x2": x2_dev_b,
                    "ent_x1": ent_x1_dev_b,
                    "ent_x2": ent_x2_dev_b,
                    "input_y": y_dev_b,
                    "add_fea": add_fea_dev_b,
                    "dropout_keep_prob": 1,
                }
                for key in ['input_x1', 'input_x2', 'ent_x1', 'ent_x2']:
                    value = feature_dict.get(key)[idx].astype(np.int32)
                    request.inputs[key].CopyFrom(tf.contrib.util.make_tensor_proto(value, shape=[1, value.size]))
                request.inputs['dropout_keep_prob'].CopyFrom(tf.contrib.util.make_tensor_proto(1.0, shape=[1]))
                request.inputs['input_y'].CopyFrom(tf.contrib.util.make_tensor_proto(1, shape=[1], dtype=np.int64))
                value = add_fea_dev_b[0].astype(np.float32)
                request.inputs['add_fea'].CopyFrom(tf.contrib.util.make_tensor_proto(value, shape=[1, value.size]))
    
                result_future = stub.Predict.future(request, 3.0)
                score = np.array(result_future.result().outputs['output'].float_val)[1]

Modify "feature_dict" according to your input variables. The variable "score" is the model output for your request instance. For more details, please refer to http://210.75.252.89:3000/hweiyi/aiLawAssistant/src/branch/master/ranking/client.py for all the codes.
> **Ref**: A detailed illustration of saving model for tensorflow-serving (https://zhuanlan.zhihu.com/p/40226973)

> **Ref**: Documents of tensorflow-serving (https://bookdown.org/leovan/TensorFlow-Learning-Notes/4-5-deploy-tensorflow-serving.html#using-tensorflow-serving-via-docker--docker--tensorflow-serving)

> **Ref**: An officail example (https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)
