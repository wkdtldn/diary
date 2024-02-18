from konlpy.tag import Okt

ok = Okt()
comment = ok.pos("댓글 데이터를 입력해주면 형태소 분석이 됩니다.")


from sklearn.model_selection import train_test_split

test_percent = 0.2 # 테스트 데이터 비율
# 훈련 데이터와 테스트 데이터 생성
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_percent)