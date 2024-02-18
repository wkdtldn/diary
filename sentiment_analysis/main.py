import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import warnings
warnings.filterwarnings('ignore')
 
#토크나이저 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
 
#모델 불러오기 
model = TFBertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states = True)
