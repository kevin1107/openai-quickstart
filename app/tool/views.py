from flask import render_template, request

from . import tool, openapi


@tool.route('/', methods=["GET", "POST"])
def index():
    return render_template('tool/index.html', **locals())

@tool.route('/code-review', methods=["GET", "POST"])
def code_review():
    if request.method == 'POST':
        query = request.form['jobDescription']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('tool/code-review.html', **locals())


@tool.route('/program-generate', methods=["GET", "POST"])
def program_generate():
    if request.method == 'POST':
        query = request.form['tweetIdeas']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('tool/program-generate.html', **locals())


@tool.route('/regex-generate', methods=["GET", "POST"])
def regex_generate():
    if request.method == 'POST':
        query = request.form['coldEmails']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('tool/regex-generate.html', **locals())


@tool.route('/sql-generate', methods=["GET", "POST"])
def sql_generate():
    if request.method == 'POST':
        query = request.form['socialMedia']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('tool/sql-generate.html', **locals())


@tool.route('/testcase-generate', methods=["GET", "POST"])
def test_case_generate():
    if request.method == 'POST':
        query = request.form['businessPitch']
        print(query)

        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

    return render_template('tool/testcase-generate.html', **locals())


@tool.route('/pdf-extract', methods=["GET", "POST"])
def pdf_extract():
    if request.method == 'POST':
        query = request.form['videoIdeas']
        print(query)
        df = openapi.pdf2dataframe()
        result = ([], '')
        result = openapi.answer_question(df, question=query.split(), debug=False)
        print(result[1])
    return render_template('tool/pdf-extract.html', **locals())
@tool.route('/xmind-transform', methods=["GET", "POST"])
def xmind_transform():
    if request.method == 'POST':
        query = request.form['videoDescription']
        print(query)
        prompt = 'AI Suggestions for {} are:'.format(query)
        openAIAnswer = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
        openapi.xls2markdown(openapi.xmind2xls())
    return render_template('tool/xmind-transform.html', **locals())
