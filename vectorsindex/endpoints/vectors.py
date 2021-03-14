import numpy as np
from flask import Blueprint, request, abort, jsonify
from injector import inject

from vectorsindex.abstract.abstract import NearestNeighborsIndex, NearestNeighborsIndexException
from vectorsindex.abstract.config import Config

vectors_blueprint = Blueprint('vectors', __name__)


@vectors_blueprint.route('/vectors', methods=['POST'])
@inject
def insert_vector(index: NearestNeighborsIndex):
    # Parse parameters
    data = request.json
    if not data:
        raise abort(400, 'No json body provided')
    # id
    try:
        vector_id = data['id']
    except KeyError:
        raise abort(400, 'Vector ID is not provided')
    except Exception:
        raise abort(400, 'ID is not valid')

    # vector
    vector = __parse_vector(data)

    # Vector insertion
    try:
        index.insert(vector_id, vector)
    except ValueError as e:
        raise abort(400, str(e))

    return 'OK', 201


@vectors_blueprint.route('/vectors/<string:vector_id>', methods=['HEAD'])
@inject
def get_vector(vector_id: str, index: NearestNeighborsIndex):
    if vector_id in index:
        return 'OK', 200

    return 'Not found', 404


@vectors_blueprint.route('/vectors/nearest-vectors', methods=['GET'])
@inject
def nearest_neighbors(index: NearestNeighborsIndex):
    # Parse parameters
    data = request.json
    if not data:
        raise abort(400, description='No json body provided')

    # Find nearest neighbors
    vector = __parse_vector(data)
    try:
        ids, distances = index.query(vector)
    except NearestNeighborsIndexException as e:
        raise abort(400, f'Query cannot be executed: {e.message}')

    response = [{'id': q[0], 'distance': float(q[1])} for q in zip(ids, distances)]
    return jsonify(response), 200


def __parse_vector(data: dict):
    try:
        vector = np.array(data['vector']).astype('float32')
    except KeyError:
        raise abort(400, description='Vector is not provided')
    except ValueError as e:
        raise abort(400, description=f'Vector is not valid: {e}')

    if Config.vectors_dimension != len(vector):
        raise abort(400, description=f'Expected vector of length {Config.vectors_dimension}')

    return vector
