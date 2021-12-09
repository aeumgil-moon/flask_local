import torch
from flask import Flask, request, render_template, jsonify
from kobart_transformers import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from konlpy.tag import Mecab
import pandas as pd
mecab = Mecab()

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
def easy_word(sentences, vocab_data):
    word_list = vocab_data['vocab']
    words = {}
    l = mecab.morphs(sentences)
    for word in l:
        if len(word)>=2:
            for i, v in enumerate(word_list):
                if word == v:
                    if not word in words:
                        words[word] = ''
                    syn = vocab_data['syn'][i]
                    if syn != words[word]:
                        words[word] += f'{syn}'
    l = []
    for key, value in words.items():
        l.append(f'{key}: {value}')
    return l

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
        
        if len(text) < 3000:
            input_ids = tokenizer.encode(text)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)

            output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            output = tokenizer.decode(output[0], skip_special_tokens=True)

            vocabs_data = pd.read_csv('./data/vocabs.csv')
            wd_list = easy_word(output,vocabs_data)
            print(wd_list)
            
            wd_key = []
            for value in wd_list:
                key_val = value.split(':')
                wd_key.append(key_val[0])

            return jsonify(result='success', result2=output, result3=wd_list, result4=wd_key)

        else:
            return jsonify(result='fail', result2='ERROR')
        
    else:
        return jsonify(result='fail', result2='ERROR')

# 404 에러
# @app.errorhandler(404)
# def page_not_found(error):
#      return render_template('page_not_found.html'), 404

if __name__ == '__main__':
    app.run(debug=True) # debug=True: 소스 수정시 Flask 재시작