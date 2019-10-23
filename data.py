import pandas as pd
import os
import tensorflow as tf 
import utils
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Dataset():
    def __init__(self, dataf , column_meta_data, exp_dir, sep= ';', batch_size= 10):
        print(dataf)
        self.data = pd.read_csv(dataf, sep = sep)
        self.numeric_column = column_meta_data['continuous']
        self.categorical_column_text = column_meta_data['categorical_txt']
        self.categorical_column_num = column_meta_data['categorical_num']
        self.bool_column = column_meta_data['bool']
        self.target = column_meta_data['target']
        self.exp_dir = exp_dir
        utils.check_dir(self.exp_dir)
        self.data_dir = os.path.join(self.exp_dir, 'data')
        self.batch_size = batch_size

    def target_label_encoding(self):
        le = LabelEncoder()
        classes = le.fit(self.data[self.target]).classes_
        target_encode_dict = {}
        for label, clas in enumerate(classes):
            target_encode_dict[clas] = label
        label_val = [int(target_encode_dict[val]) for val in self.data[self.target].values]
        self.data[self.target]=  label_val

    def split_dataset(self):
        self.target_label_encoding()
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.15)
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.15)
        utils.check_dir(self.data_dir)
        print(self.train_data.shape[0])
        train_len_size = int(self.train_data.shape[0] / 100) * 100
        print(train_len_size)
        self.train_data = self.train_data.sample(train_len_size)
        print(self.val_data.shape[0])
        val_len_size = int(self.val_data.shape[0] / 100) * 100
        print(val_len_size)
        self.val_data = self.val_data.sample(val_len_size)
        print(f"Train data shape - {self.train_data.shape}")
        print(f"Test data shape - {self.test_data.shape}")
        print(f"Val data shape - {self.val_data.shape}")
        self.train_data.to_csv(f"{self.data_dir}/train.csv", index=False)
        self.test_data.to_csv(f"{self.data_dir}/test.csv", index=False)
        self.val_data.to_csv(f"{self.data_dir}/val.csv", index=False)

    def handle_null(self, dataF):
        for col in dataF:
            if col in self.numeric_column:
                fill_val = 0.0 
            elif col in self.categorical_column_text:
                fill_val = "null"
            elif col in self.categorical_column_num or self.bool_column:
                fill_val = 0.9

            if (int(dataF[col].isnull().values.sum())) > 0:
                if col != self.target:
                    dataf[col] = dataF[col].fill_na(fill_val, inplace=False)
                else:
                    dataF = dataF.dropna(subset = [self.target])
        return dataF

    def load_dataset(self):

        def df_process(dataF, batch_size, shuffle = False):
            dataF = dataF.copy()
            #print(dataF.columns)
            labels = dataF.pop(self.target)
            labels = tf.one_hot(labels, 2)
            ds = tf.data.Dataset.from_tensor_slices((dict(dataF), labels))
            if shuffle:
                ds = ds.shuffle(buffer_size = len(dataF))
            ds = ds.batch(batch_size)
            return ds

        train_data = self.handle_null(self.train_data)
        test_data = self.handle_null(self.test_data)
        val_data = self.handle_null(self.val_data)

        train_ds = df_process(train_data, batch_size = self.batch_size, shuffle=True)
        test_ds = df_process(test_data, batch_size = self.batch_size, shuffle= False)
        val_ds = df_process(val_data, batch_size = self.batch_size, shuffle = False)

        return( train_ds, test_ds, val_ds)
    def make_feature_layer(self):
        feature_cols = []

        for col in self.numeric_column:
            feature_cols.append(feature_column.numeric_column(col))

        for col in self.categorical_column_num:
            unique_count = self.data[col].nunique()
            feat_cols = feature_column.embedding_column(
                feature_column.categorical_column_with_hash_bucket(
                    col, hash_bucket_size=int(3*unique_count)), dimension=1)
            feature_cols.append(feat_cols)

        for col in self.categorical_column_text:
            unique_count = self.data[col].nunique()
            feat_cols = feature_column.embedding_column(
                feature_column.categorical_column_with_hash_bucket(
                    col, hash_bucket_size=int(3*unique_count)), dimension=1)
            feature_cols.append(feat_cols)
      
        for col in self.bool_column:
            unique_count = self.data[col].nunique()
            feat_cols = feature_column.embedding_column(
                feature_column.categorical_column_with_hash_bucket(
                    col, hash_bucket_size=3), dimension=1)
            feature_cols.append(feat_cols) 
        return feature_cols
