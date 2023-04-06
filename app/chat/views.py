from flask import redirect, render_template, request, url_for

from . import chat, api


@chat.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        animal = request.form["animal"]
        answer = api.generateChatResponse(animal)
        return redirect(url_for("chat.index", result=answer))
    result = request.args.get("result")
    return render_template("chat/index.html", result=result)
