from flask import render_template, jsonify, request

from . import main, api


@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        res = {}
        res['answer'] = api.generateChatResponse(prompt)
        return jsonify(res), 200
    return render_template('index.html', **locals())
