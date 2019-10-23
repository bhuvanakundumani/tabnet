import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from data import Dataset
from model import TabNet


col_metadata_file = 'col_metadata.pkl'
input_file = "bank-full.csv"
exp_dir= "test_dir"
batch_size= 10
sparsity_loss_weight= 0.0001
decay_rate = 0.95
decay_every = 500
init_localearning_rate = 0.02

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

col_metadata = pkl.load(open(col_metadata_file, 'rb'))

data_preprocess = Dataset(input_file, col_metadata, sep=';',exp_dir= exp_dir, batch_size= 100)

data_preprocess.split_dataset()

train_data, test_data, val_data = data_preprocess.load_dataset()

inp_feature_columns = data_preprocess.make_feature_layer()

MODEL = TabNet(feature_columns = inp_feature_columns, num_features= 16, feature_dim = 128, output_dim= 64, num_decision_steps= 6, relaxation_factor= 1.5, virtual_batch_size= 10, num_classes=2, batch_size= 100, batch_momentum= 0.7, is_training= True)

global_step = tf.compat.v1.train.get_or_create_global_step()

learning_rate = tf.compat.v1.train.exponential_decay(
                init_localearning_rate,
                global_step= global_step,
                decay_steps= decay_every,
                decay_rate= decay_rate
)


optimizer= tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate)

'''
def model_objective(entropy, actual_output, predicted_output):
    softmax_orig_key_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits= predicted_output, labels=  actual_output
            )
    )
    train_loss = softmax_orig_key_op + sparsity_loss_weight * entropy
    return train_loss

@tf.function()
def training_steps(tabnet_model, inp_data, out_data, k= 1, batch_size= 10):
    print("Train step called")
    for i in range(k):
        print(i)
        with tf.GradientTape() as model_tape:
           logits, predictions, entropy = MODEL(inp_data)
           print("##one block finished")
           total_loss = model_objective(entropy, out_data, logits)
           print("##second block finished")
           model_gradient = model_tape.gradient(total_loss, tabnet_model.trainable_variables)
           print("## third block finished")
           optimizer.apply_gradients(zip(model_gradient, tabnet_model.trainable_variables))


def training(dataset, epochs):
    for epoch in range(epochs):
        for i, (inp_data, out_data) in enumerate(dataset):
            print(f'{i}-iteration')
            training_steps(MODEL, inp_data, out_data)
training(train_data, 2)
'''
history = MODEL.compile(optimizer, loss= "categorical_crossentropy", metrics= ['accuracy'])
print(history.history)
MODEL.fit(train_data, epochs= 5, validation_data= val_data)
# Evaluating on val_data
results = MODEL.evaluate(val_data)
print(results)

print("Predictions on the first three elements in the dataset")
predictions = MODEL.predict(test_data[:3])
print(f"predictions shape is {preditions.shape}")
print(f"predictions are {predictions}")
