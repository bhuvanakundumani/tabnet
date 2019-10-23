import pickle
column_data = {
    'continuous': ['age','balance','duration','campaign','pdays','previous','day'],
    'categorical_txt':['job','marital','education','housing','poutcome','contact'],
    'categorical_num':['month'],
    'target':'y',
    'bool':['loan', 'default']
}
pickle_out = open("col_metadata_test.pkl", "wb")
pickle.dump(column_data, pickle_out)
pickle_out.close()


pickle_in = open("col_metadata_test.pkl", "rb")
col_data_frm_pickle = pickle.load(pickle_in)
print(f"Here is your column data from pickle {col_data_frm_pickle}")


# Alternative way !!

col_data_from_pickle1 = pickle.load(open("col_metadata_test.pkl", "rb"))
print(f"\n\nAnother alternative way is {col_data_from_pickle1}")
