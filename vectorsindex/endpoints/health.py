from flask import Blueprint, jsonify
from injector import inject

from vectorsindex.abstract.abstract import NearestNeighborsIndex
from vectorsindex.abstract.config import Config

health_blueprint = Blueprint('health', __name__)


@health_blueprint.route('/__health', methods=['GET'])
@inject
def health(index: NearestNeighborsIndex):
    stored_vectors = len(index)
    data = {
        'storedVectors': stored_vectors,
        'storedVectorsLimit': Config.stored_vectors_limit
    }

    return jsonify(data), 200
