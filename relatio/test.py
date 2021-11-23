from utils import split_into_sentences
import pandas as pd 

test_string = {"doc" : "Some text. Some more text. Some more and more text.", "id" : 0}
test_bool = {"doc" : False, "id" : 0}
test_int = {"doc" : 10, "id" : 0}
test_float = {"doc" : 10.000, "id" : 0}

df_teststring, df_testbool = pd.DataFrame([test_string]), pd.DataFrame([test_bool])
df_testint, df_testfloat =  pd.DataFrame([test_int]), pd.DataFrame([test_float])
print(split_into_sentences(df_teststring))
print(split_into_sentences(df_testbool))
print(split_into_sentences(df_testint))
print(split_into_sentences(df_testfloat))

