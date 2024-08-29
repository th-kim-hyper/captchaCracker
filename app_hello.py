import sys
import os
import webbrowser
from threading import Timer
from flask import Flask, render_template
from .. import util
     
#https://github.com/smoqadam/PyFladesk/issues/9
#template의 directory를 불러오도록 하는 코드'''
if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    app = Flask(__name__, template_folder=template_folder)
else:
    app = Flask(__name__)
     
@app.route('/')
@app.route('/<name>')
def hello_world(name=None):
    return render_template('index_hello.html', name=name)
     
def open_browser():
      webbrowser.open_new('http://127.0.0.1:'+str(port)+'/') 
#https://stackoverflow.com/questions/54235347/open-browser-automatically-when-python-code-is-executed
#세션 종료 후가 아니라 코드 실행 후 자동적으로 웹 페이지가 열리도록 하기 위하여
     
port=9000
             
if __name__ == "__main__":
    Timer(1,open_browser).start();
    app.run(port=port, debug=True)      