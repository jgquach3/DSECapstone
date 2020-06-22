from flask import Blueprint, jsonify, make_response, request
from .models import Topic

topic = Blueprint('topic', __name__)

@topic.route('/api/topic', methods=['GET'])
def get_topic():
    """
    should just expose queryable params via query string
    and then have logic here to interpret it
    """

    limit = request.args.get('limit', default=100)

    try:
        topic_query = Topic.query
    except Exception as e:
        return make_response(jsonify(e), 500)

    results = topic_query.limit(limit).all()

    try:
        results = Topic.serialize_list(results)
    except Exception as e:
        results = results.serialize()

    try:
        return make_response(jsonify(results), 200)
    except Exception as e:
        return make_response(jsonify(e), 500)


@topic.route('/api/topic/<topic_id>', methods=['GET'])
def get_topic_by_id(topic_id):
    """
    Get single topic desc
    """

    # using limit param here as a just in case
    limit = request.args.get('limit', default=100)

    results = Topic.query.filter_by(id=topic_id).limit(limit).all()

    try:
        results = Topic.serialize_list(results)
    except Exception as e:
        results = results.serialize()

    try:
        return make_response(jsonify(results), 200)
    except Exception as e:
        return make_response(jsonify(e), 500)
   	