import pandas as pd
columns = ['id','text','label']
PATH = 'C:/workplaces/wkdtldn/diary/test/data/'
train_data = pd.read_csv(PATH + 'ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna() 
test_data = pd.read_csv(PATH + 'ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

train_data.to_csv('data/train_data.csv',index=False)
test_data.to_csv('data/test_data.csv',index=False)


from konlpy.tag import Mecab
mecab = Mecab()
print(mecab.morphs('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))