from flask import Flask, request, render_template

# __name__ 은 파이썬에서 내부적으로 사용하는 특별한 변수 입니다.
# 보통 __name__의 값을 출력 하면 모듈의 이름을 뜻하는데, 만약 실행이 되는 .py 모듈은 __main__ 으로 나옵니다.
# 단일 모듈: app = Flask(__name__)
# 패키지 형태: app = Flask('application 명 지정') // ex) app = Flask('infer_flask')
app = Flask(__name__) # 전역객체로 사용(인스턴스 선언)

# app의 객체의 route함수에 request 인자를 넘기면서 HTTP요청을 처리 합니다.
# 데코레이터: 파이썬에서 기본적으로 사용되는 기술로, 함수내의 추가적인 작업들을 간단하게 사용할 수 있도록 도와주는 기술
# 데코레이터 URL과 함수를 연결
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/method', methods=['GET', 'POST'])
def method():
    # GET은 request.args
    if request.method == 'GET':
        num = request.args['num'] # 1번 방법
        name = request.args.get('name') # 2번 방법
        return f'GET으로 전달된 데이터: {num} / {name}'
    # POST는 request.form
    else:
        num = request.form['num'] # 1번 방법
        name = request.form.get('name') # 2번 방법
        return f'POST로 전달된 데이터: {num} / {name}'

if __name__ == '__main__':
    app.run(debug=True) # debug=True: 소스 수정시 Flask 재시작