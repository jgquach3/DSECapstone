from flask import Blueprint, jsonify, make_response, request

from api.database import db
from .models import NewsArticle
from .models import OpioidDocLoc
from .models import TopicDoc

newsarticle = Blueprint('newsarticle', __name__)

#### global being used to manually serialize keys 
key_list = ['city', 'county', 'state']

@newsarticle.route('/api/newsarticle', methods=['GET'])
def get_newsarticle():
    """
    should just expose queryable params via query string
    and then have logic here to interpret it
    """

    before_collection_date = request.args.get('before_collection_date')
    after_collection_date = request.args.get('after_collection_date')
    before_publish_date = request.args.get('before_publish_date')
    after_publish_date = request.args.get('after_publish_date')
    language = request.args.get('languge')
    newsindex = request.args.get('newsindex')
    keywords = request.args.get('keywords')
    author = request.args.get('author')

    location_key = request.args.get('location_key')
    location_value = request.args.get('location_value')

    topic_key = request.args.get('topic_key')

    # if true means, location is required
    location = request.args.get('location')

    # have a list of default columns?

    # maybe should have a flag to get location information?
    # or should always just include????
    columns = request.args.get('columns')

    limit = request.args.get('limit', default=500)

    try:
        news_query = db.session.query(NewsArticle, OpioidDocLoc, TopicDoc)

        if location and location.lower() == 'true':
            news_query = news_query.join(OpioidDocLoc, NewsArticle.id == OpioidDocLoc.docid)
        else:
            news_query = news_query.join(OpioidDocLoc, NewsArticle.id == OpioidDocLoc.docid, isouter=True)

        if before_collection_date:
            news_query = news_query.filter(NewsArticle.collectiondate <= before_collection_date)
        if after_collection_date:
            news_query = news_query.filter(NewsArticle.collectiondate >= after_collection_date)
        if before_publish_date:
            news_query = news_query.filter(NewsArticle.publishdate <= before_publish_date)
        if after_publish_date:
            news_query = news_query.filter(NewsArticle.publishdate >= after_publish_date)
        if language:
            news_query = news_query.filter(NewsArticle.language == language)
        if newsindex:
            news_query = news_query.filter(NewsArticle.newsindex == newsindex)
        if keywords:
            keywords = keywords.split(',')
            news_query = news_query.filter(NewsArticle.keywords.contains(keywords))
        if author:
            author = author.split(',')
            news_query = news_query.filter(NewsArticle.keywords.contains(author))

        if location_key:

            # currently none is being interpreted as actually none and not None
            if location_key == 'state':
                news_query = news_query.filter(OpioidDocLoc.state != ('none' or None))
                if location_value:

                    news_query = news_query.filter(OpioidDocLoc.state == location_value)

            elif location_key == 'county':
                news_query = news_query.filter(OpioidDocLoc.county != ('none' or None))
                if location_value:
                    location_value = location_value.split(',')

                    news_query = news_query.filter(OpioidDocLoc.county == location_value[0])

                    if len(location_value) > 1:
                        news_query = news_query.filter(OpioidDocLoc.state == location_value[1])


            elif location_key == 'city':
                news_query = news_query.filter(OpioidDocLoc.city != ('none' or None))
                if location_value:
                    location_value = location_value.split(',')

                    news_query = news_query.filter(OpioidDocLoc.city == location_value[0])

                    if len(location_value) > 1:
                        news_query = news_query.filter(OpioidDocLoc.state == location_value[1])

        if topic_key:
            news_query = news_query.join(TopicDoc, NewsArticle.id == TopicDoc.docid).filter(TopicDoc.topic == topic_key)


        news_query = news_query.distinct(NewsArticle.id)


    except Exception as e:
        return make_response(jsonify(e), 500)
    
    results = news_query.limit(limit).all()

    result_set = []
    for r in results:

        news_model = r[0].serialize()

        if r[1]:
            loc_model = r[1].serialize()

            for k in key_list:
                news_model[k] = loc_model[k]
        else:
            for k in key_list:
                news_model[k] = None

        if columns:
            news_model = { key: news_model[key] for key in columns.split(',') }

        result_set.append(news_model)

    return make_response(jsonify(result_set), 200)


@newsarticle.route('/api/newsarticle/<news_id>', methods=['GET'])
def get_newsarticle_by_id(news_id):
    """
    Get single news article
    """

    """
    try:
        news_obj = db.session.query(NewsArticle)\
        .filter(NewsArticle.id == news_id)\
        .all()

        return make_response(jsonify(news_obj[0].serialize()), 200)

    except Exception as e:
        print (e)
        return make_response(jsonify(e), 500)
    """

    try:
        news_obj = db.session.query(NewsArticle, OpioidDocLoc)\
        .join(OpioidDocLoc, NewsArticle.id == OpioidDocLoc.docid, isouter=True)\
        .filter(NewsArticle.id == news_id)\
        .all()

    except Exception as e:
        print (e)
        return make_response(jsonify(e), 500)

    if news_obj:

        for r in news_obj:

            news_model = r[0].serialize()

            if r[1]:
                loc_model = r[1].serialize()

                for k in key_list:
                    news_model[k] = loc_model[k]

            else:
                for k in key_list:
                    news_model[k] = None

        return make_response(jsonify(news_model), 200)

    else:
        return make_response(jsonify({}), 200)
