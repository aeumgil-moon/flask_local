import torch
from flask import Flask, request, render_template, jsonify
from kobart_transformers import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

app = Flask(__name__)

## 모델 로드 ##
def load_model():
    model = BartForConditionalGeneration.from_pretrained(
        "D:/project/flask_local/KoBART-summarization/kobart_summary" # 애움길 KoBARTSum모델로 설정
        )
    # tokenizer = get_kobart_tokenizer()
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()

## 데이터 전처리 ##
def text_eda(text):
    text = text.replace('\n', '')
    text = text.strip()

    return text

@app.route('/')
def index():
    return render_template('aeumgil.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # print(data)
    # print(data['text'])
    text = data['text']

    if text:
        text = text_eda(text)
        # print(text)
        # print(len(text))
        
        if len(text) < 3000:
            input_ids = tokenizer.encode(text)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)

            ## 1. 샘플 모델 ##
            # output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)

            ## 2. top_k & top_p sampling
            ## 참고: Transformer로 텍스트를 생성하는 다섯 가지 전략(https://littlefoxdiary.tistory.com/46)
            # output = model.generate(
            #     input_ids,
            #     do_sample=True, # 샘플링 사용
            #     max_length=100,
            #     top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
            #     top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
            #     # num_beams=5, # beam search 사용(1보다 큰 값 사용)
            #     # early_stopping=True, #EOS토큰이 나오면 생성을 중단
            #     # no_repeat_ngram_size=2, # n-gram에서 n만큼의 어구가 반복되지 않도록 설정함 // 이렇게 설정하면 해당 단어는 n-gram에서 설정한 n보다 작은 값의 빈도 수 만큼 출력된다.
            #     num_return_sequences=3 # n개의 문장을 리턴 // no_repeat_ngram_size의 단점을 보완하기 위해 설정(num_beams보다 작거나 같아야한다.)
            #     )

            ## 3. beam search
            output = model.generate(
                input_ids,
                max_length=100,
                num_beams=5, # beam search 사용(1보다 큰 값 사용)
                early_stopping=True, #EOS토큰이 나오면 생성을 중단
                no_repeat_ngram_size=3, # n-gram에서 n만큼의 어구가 반복되지 않도록 설정함 // 이렇게 설정하면 해당 단어는 n-gram에서 설정한 n보다 작은 값의 빈도 수 만큼 출력된다.
                num_return_sequences=3 # n개의 문장을 리턴 // no_repeat_ngram_size의 단점을 보완하기 위해 설정(num_beams보다 작거나 같아야한다.)
                )
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(output)
            # print(type(output))

            return jsonify(result='success', result2=output)

        else:
            return jsonify(result='fail', result2='ERROR')
        
    else:
        return jsonify(result='fail', result2='ERROR')

if __name__ == '__main__':
    app.run(debug=True) # debug=True: 소스 수정시 Flask 재시작