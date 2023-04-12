from flask_marshmallow import Schema
from marshmallow.fields import Str


class WelcomeModel:
    def __init__(self):
        self.message = "Hello World!"


class WelcomeSchema(Schema):
    class Meta:
        # Fields to expose
        fields = ["message"]

    message = Str()
