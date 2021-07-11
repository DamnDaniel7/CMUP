from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from sentiment import Sentiment 
import speedtest

app = Flask(__name__)
nlk = Sentiment()

cors = CORS(app)

@app.route('/', methods=['POST'])
def hello() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"])
    print(res)
    return jsonify(res)

@app.route('/speed', methods=['POST'])
def speed() -> Response:
    s = speedtest.Speedtest()
    s.get_servers()
    s.get_best_server()
    s.download()
    s.upload()
    s = s.results.dict()
    res = {"download": round((s["download"]/1024)/1024), "upload": round((s["upload"]/1024)/1024), "ping": round(s["ping"]) }
    #res = {"download": 170, "upload": 80, "ping": 10 }
    print(res)
    return jsonify(res)

@app.route('/who', methods=['POST'])
@app.route('/home', methods=['POST'])
def who_home() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"],"home")
    return jsonify(res)

@app.route('/dark', methods=['POST'])
def light_on() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"],"dark")
    return jsonify(res)

@app.route('/bright', methods=['POST'])
def light_off() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"],"bright")
    return jsonify(res)

@app.route('/loud', methods=['POST'])
def volume_down() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"],"loud")
    return jsonify(res)

@app.route('/quiet', methods=['POST'])
def volume_up() -> Response:
    data = request.get_json()
    res = nlk.pipeline(data["texto"],"quiet")
    return jsonify(res)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)