import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, DenseFeatures
from utils import glu_layer
import tensorflow_addons as tfa

class TabNet(Model):
    def __init__(self, feature_columns, num_features, feature_dim, output_dim, num_decision_steps, relaxation_factor, virtual_batch_size,
                        num_classes, batch_size, batch_momentum, is_training,epsilon=0.00001):
        super().__init__(name="TabNet")
        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.relaxation_factor = relaxation_factor
        self.is_training= is_training
        #Encoder layers
        self.input_feature_layers = DenseFeatures(self.feature_columns)
        self.input_batch_norm = BatchNormalization(momentum= self.batch_momentum)

        #Decision step dependent variables
        self.output_aggregated =  tf.zeros([self.batch_size, self.output_dim])
        self.mask_values = tf.zeros([self.batch_size, self.num_features])
        self.aggregated_mask_values = tf.zeros([self.batch_size, self.num_features])
        self.complementary_aggregated_mask_values = tf.ones([self.batch_size, self.num_features])

        self.total_entropy = 0

        if self.is_training:
            self.v_b = self.virtual_batch_size
        else:
            self.v_b = 1
        
        self.transformer_layers = []
        self.attentive_layers = []
        for step in range(self.num_decision_steps):
            step_layers = []
            attentive_step_layers = []
            step_layers.append(Dense(units= self.feature_dim * 2, use_bias= False))
            step_layers.append(BatchNormalization(momentum= self.batch_momentum, virtual_batch_size= self.v_b))
            #put glu layer
            step_layers.append(Dense(units= self.feature_dim * 2, use_bias= False))
            step_layers.append(BatchNormalization(momentum= self.batch_momentum, virtual_batch_size = self.v_b))
            #glu layer and previous residue
            step_layers.append(Dense(units= self.feature_dim * 2, use_bias= False))
            step_layers.append(BatchNormalization(momentum= self.batch_momentum, virtual_batch_size = self.v_b))
            #glu layer and previous residue
            step_layers.append(Dense(units= self.feature_dim * 2, use_bias= False))
            step_layers.append(BatchNormalization(momentum= self.batch_momentum, virtual_batch_size = self.v_b))
            self.transformer_layers.append(step_layers)
            #attentive_transformer layers
            attentive_step_layers.append(Dense(units = self.num_features,use_bias = False))
            attentive_step_layers.append(BatchNormalization( momentum= self.batch_momentum, virtual_batch_size= self.v_b))
            self.attentive_layers.append(attentive_step_layers)

        self.classification_layers =  Dense(units = self.num_classes,activation='softmax',  use_bias = False)
    
   # @tf.function
    def call(self, data):
        feature_data= self.input_feature_layers(data)
        feature_out= self.input_batch_norm(feature_data)
        masked_features = feature_out
        
        #to_do check re_use flag in dense
        for step in range(self.num_decision_steps):
            #tranformer1_block
            transformer1_output = self.transformer_layers[step][0](masked_features)
            #print(transformer1_output.shape)
            transformer1_output = self.transformer_layers[step][1](transformer1_output, training = self.is_training)
            #print(transformer1_output.shape)
            transformer1_output = glu_layer(transformer1_output, self.feature_dim)
            #transformer2_block
    
            #print("######",transformer1_output.shape)
            transformer2_output = self.transformer_layers[step][2](transformer1_output)
            transformer2_output = self.transformer_layers[step][3](transformer2_output, training = self.is_training)
            transformer2_output = glu_layer(transformer2_output, self.feature_dim) + transformer1_output * np.sqrt(0.5)
            #transformer3_block
            transformer3_output = self.transformer_layers[step][4](transformer2_output)
            transformer3_output = self.transformer_layers[step][5](transformer3_output, training = self.is_training)
            transformer3_output = glu_layer(transformer3_output, self.feature_dim) + transformer2_output * np.sqrt(0.5)
            #transformer4_block
            transformer4_output = self.transformer_layers[step][6](transformer3_output)
            transformer4_output = self.transformer_layers[step][7](transformer4_output)
            transformer4_output = glu_layer(transformer4_output, self.feature_dim) + transformer3_output * np.sqrt(0.5)

        
            if step > 0:
                decision_out = tf.nn.relu(transformer4_output[:, :self.output_dim])
                #decision_aggregation
                self.output_aggregated += decision_out
                #feature_importance attribute
                self.scale_agg = tf.reduce_sum(decision_out, axis= 1, keepdims = True) / (self.num_decision_steps - 1)
                self.aggregated_mask_values += mask_values * self.scale_agg

            features_for_coef = (transformer4_output[:, self.output_dim:])

            if step < self.num_decision_steps - 1:
                mask_values = self.attentive_layers[step][0](features_for_coef)
                mask_values = self.attentive_layers[step][1](mask_values)

                mask_values *= self.complementary_aggregated_mask_values
                mask_values =  tfa.layers.sparsemax.Sparsemax()(mask_values)
                self.complementary_aggregated_mask_values *= (self.relaxation_factor - mask_values)

                self.total_entropy += tf.reduce_mean(tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon),
                                                                axis = 1)) / (self.num_decision_steps - 1)
                #feature selection
                masked_features = tf.multiply(mask_values, feature_out)
                #need to insert tensorgraph visualization for mask_values
            else:
                self.total_entropy = 0.
            
        #classification part
        self.add_loss(0.0001 * self.total_entropy)
        logits = self.classification_layers(self.output_aggregated)
        #predictions = tf.nn.softmax(logits)
        #return tf.argmax(logits,axis=1)
        return logits
