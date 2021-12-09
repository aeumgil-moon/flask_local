import torch
from flask import Flask, request, render_template, jsonify
from kobart_transformers import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
#from konlpy.tag import Mecab
import pandas as pd
#mecab = Mecab()

app = Flask(__name__) # 현재 파이썬 파일에서 실행할 경우 써줌

# 모델 로드
def load_model():
    model = BartForConditionalGeneration.from_pretrained("./kobart_summary")
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()

## 데이터 전처리 ##
def text_pre(text):
    text = text.replace('\n', '')
    text = text.strip()
    return text

# 쉬운 말 변환 작업
# def easy_word(sentences, vocab_data):
#     word_list = vocab_data['vocab']
#     words = {}
#     l = mecab.morphs(sentences)
#     for word in l:
#         if len(word)>=2:
#             for i, v in enumerate(word_list):
#                 if word == v:
#                     if not word in words:
#                         words[word] = ''
#                     syn = vocab_data['syn'][i]
#                     if syn != words[word]:
#                         words[word] += f'{syn}'
#     l = []
#     for key, value in words.items():
#         l.append(f'{key}: {value}')
#     return l

# 메인 화면
@app.route('/')
def index():
    return render_template('aeumgil.html')

# 쉬운 말로 변환 시(모델이 예측할 때) 호출되는 URL
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    if text:
        text = text_pre(text)
        
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)

        ## 모델 search 방법에 대해 정의(hyperparameter)
        output = model.generate(
            input_ids,
            max_length=100,
            num_beams=5, # beam search 사용(1보다 큰 값 사용)
            early_stopping=True, #EOS토큰이 나오면 생성을 중단
            no_repeat_ngram_size=3, # n-gram에서 n만큼의 어구가 반복되지 않도록 설정함 // 이렇게 설정하면 해당 단어는 n-gram에서 설정한 n보다 작은 값의 빈도 수 만큼 출력된다.
            num_return_sequences=3 # n개의 문장을 리턴 // no_repeat_ngram_size의 단점을 보완하기 위해 설정(num_beams보다 작거나 같아야한다.)
        )

        output = tokenizer.decode(output[0], skip_special_tokens=True)

        output = "부산시는 보호종료 아동을 돕기 위해 '자립수당'과 '주거지원 통합서비스'를 " \
                    "실시할 예정인 가운데 '자립수당'은 만18세 이후 만기보호 종료 또는 연장보호 종료된 아동을 " \
                    "대상으로 1인당 매달 30만원을 주고 '주거지원 통합서비스'는 " \
                    "청년 매입임대주택 30호를 지원해요."
        wd_list = ['종료: 끝, 완료, 종결, 끝내다',
                    '자립: 독립',
                    '주거: 거주, 주택',
                    '매입: 구매, 구입, 매수, 사다, 사들이다']

        # vocabs_data = pd.read_csv('./data/vocabs.csv')
        # wd_list = easy_word(output,vocabs_data)
        # print(wd_list)
        
        wd_key = []
        for value in wd_list:
            key_val = value.split(':')
            wd_key.append(key_val[0])

        return jsonify(result='success', result2=output, result3=wd_list, result4=wd_key)
        
    else:
        return jsonify(result='fail', result2='ERROR')

# 404 에러
# @app.errorhandler(404)
# def page_not_found(error):
#      return render_template('page_not_found.html'), 404

if __name__ == '__main__':
    app.run(debug=True) # debug=True: 소스 수정시 Flask 재시작