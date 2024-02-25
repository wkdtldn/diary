class Tokenizer:
    def __init__(self,text):
        self.text = text

    # 특수 문자 제거
    @staticmethod
    def RemoveSpecials(text):
        wrap_flat = {"state" : False}
        filtering_text = list()
        wrapping_text = list()
        specials = ['!','`','@','#','$','%','^',"&",'*','(',')','-','_','|','+','\'','{','}','[',']',';',':','"',"'",',','.','<','>','?','/']
        for word in text:
            for special in specials:
                if word == special:
                    if special == '(' or special == '[' or special == '{' or special == '<':
                        wrap_flat["state"] = True
                    else:
                        filtering_text.append(word)
            if wrap_flat["state"] == True:
                wrapping_text.append(word)

text = "안녕하세요 <Hello>!?:"

Tokenizer.RemoveSpecials(text)