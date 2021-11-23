from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return 'MAIN'

@app.route('/home')
def home():
    return 'Hello, World'

@app.route('/user')
def user():
    return 'This is User'

@app.route('/user/<user_name>/<string:user_id>')
def user_variable(user_name, user_id):
    return f'#### Hello, {user_name}({user_id}) ####'

if __name__ == '__main__':
    app.run(debug=True) # debug=True: 소스 수정시 Flask 재시작