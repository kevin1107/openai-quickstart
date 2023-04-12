from http import HTTPStatus

from flasgger import swag_from
from flask import jsonify, request, g, url_for

from app.api.welcome import WelcomeModel
from app.api.welcome import WelcomeSchema
from . import api


@api.route('/')
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Welcome to the Flask Starter Kit',
            'schema': WelcomeSchema
        }
    }
})
def welcome():
    """
    1 liner about the route
    A more detailed description of the endpoint
    ---
    """
    result = WelcomeModel()
    return WelcomeSchema().dump(result), 200


@api.route('/files')
def get_files():
    page = request.args.get('page', 1, type=int)
    pagination = {}
    files = pagination.items
    prev = None
    if pagination.has_prev:
        prev = url_for('api.get_files', page=page - 1)
    next = None
    if pagination.has_next:
        next = url_for('api.get_files', page=page + 1)
    return jsonify({
        'files': [post.to_json() for post in files],
        'prev': prev,
        'next': next,
        'count': pagination.total
    })

@api.route('/files', methods=['POST'])
def new_post():
    post = {}
    post.author = g.current_user
    return jsonify(post.to_json()), 201, \
        {'Location': url_for('api.get_post', id=post.id)}


@api.route('/files/<int:id>', methods=['PUT'])
def edit_post(id):
    post = {}
    post.body = request.json.get('body', post.body)
    return jsonify(post.to_json())
