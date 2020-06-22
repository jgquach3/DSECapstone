from flask import Blueprint, jsonify, make_response, request
from .models import TopicDoc

topicdoc = Blueprint('topicdoc', __name__)

@topicdoc.route('/api/topicdoc', methods=['GET'])
def get_topicdoc():
    """
    should just expose queryable params via query string
    and then have logic here to interpret it
    """

    before_model_date = request.args.get('before_model_date')
    after_model_date = request.args.get('after_model_date')
    before_prob = request.args.get('less_prob')
    after_prob = request.args.get('greater_prob')
    topic = request.args.get('topic')

    limit = request.args.get('limit', default=100)

    try:
        topic_query = TopicDoc.query

        if before_model_date:
            topic_query = topic_query.filter(TopicDoc.modeldate <= before_model_date)
        if after_model_date:
            topic_query = topic_query.filter(TopicDoc.modeldate >= after_model_date)
        if before_prob:
            topic_query = topic_query.filter(TopicDoc.prob <= before_prob)
        if after_prob:
            topic_query = topic_query.filter(TopicDoc.prob >= after_prob)
        if topic:
            topic_query = topic_query.filter_by(topic=topic)
    except Exception as e:
        return make_response(jsonify(e), 500)

    results = topic_query.limit(limit).all()

    try:
        results = TopicDoc.serialize_list(results)
    except Exception as e:
        results = results.serialize()

    return make_response(jsonify(results), 200)


@topicdoc.route('/api/topicdoc/<doc_id>', methods=['GET'])
def get_topicdoc_by_id(doc_id):
    """
    Get single topic article
    """

    # using limit param here as a just in case
    latest = request.args.get('latest')
    limit = request.args.get('limit', default=100)

    if latest and latest.lower() == 'true':
        try:
            topic_obj = TopicDoc.query.filter_by(docid=doc_id)\
            .order_by(TopicDoc.modeldate.desc())\
            .order_by(TopicDoc.prob.desc()).first()

            return make_response(jsonify(TopicDoc.serialize(topic_obj)), 200)
        except Exception as e:

            print (e)
            return make_response(jsonify(e), 500)


    try:
        topic_obj = TopicDoc.query.filter_by(docid=doc_id).limit(limit).all()
        return make_response(jsonify(TopicDoc.serialize_list(topic_obj)), 200)
    except Exception as e:
        return make_response(jsonify(e), 500)
   	