from flask import Flask
from flasgger import Swagger
from flask_bootstrap import Bootstrap

from config import config

bootstrap = Bootstrap()


def create_app(config_name):
    app = Flask(__name__)

    app.config['SWAGGER'] = {
        'title': 'Flasgger RESTful',
        # 'uiversion': 2
    }
    swagger = Swagger(app)

    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    bootstrap.init_app(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .chat import chat as chat_blueprint
    app.register_blueprint(chat_blueprint, url_prefix='/chat')

    from .tool import tool as chat_blueprint
    app.register_blueprint(chat_blueprint, url_prefix='/tool')

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')

    return app
