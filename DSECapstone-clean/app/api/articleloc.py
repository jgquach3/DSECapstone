from flask import Blueprint, jsonify, make_response, request

from sqlalchemy import func

from api.database import db

from datetime import datetime, timedelta
import requests

from .models import NewsArticle
from .models import TopicDoc
from .models import OpioidDocLoc

articleloc = Blueprint('articleloc', __name__)

@articleloc.route('/api/articleloc/_stats/count', methods=['GET'])
def get_group_loc_stats():
    """
    group by each key and return the respective values
    and you can filter stuff too
    """

    before_collection_date = request.args.get('before_collection_date')
    after_collection_date = request.args.get('after_collection_date')
    before_publish_date = request.args.get('before_publish_date')
    after_publish_date = request.args.get('after_publish_date')

    location_key = request.args.get('location_key')
    topic_key = request.args.get('topic_key')
    keywords = request.args.get('keywords')

    nones = request.args.get('none')

    if location_key not in ['city', 'county', 'state']:
        return make_response(jsonify([]), 200)

    try:

        news_query = db.session.query(NewsArticle.id)

        joiner = False
        if before_collection_date:
            news_query = news_query.filter(NewsArticle.collectiondate <= before_collection_date)
            joiner = True
        if after_collection_date:
            news_query = news_query.filter(NewsArticle.collectiondate >= after_collection_date)
            joiner = True
        if before_publish_date:
            news_query = news_query.filter(NewsArticle.publishdate <= before_publish_date)
            joiner = True
        if after_publish_date:
            news_query = news_query.filter(NewsArticle.publishdate >= after_publish_date)
            joiner = True
        if keywords:
            keywords = keywords.split(',')
            news_query = news_query.filter(NewsArticle.keywords.contains(keywords))
            joiner = True
        if topic_key:
            news_query = news_query.join(TopicDoc, NewsArticle.id == TopicDoc.docid).filter(TopicDoc.topic == topic_key).distinct(NewsArticle.id)
            joiner = True


        if joiner:
            news_sub_query = news_query.subquery()

            if location_key == 'city':
                stats_query = db.session.query(OpioidDocLoc.city, func.count(OpioidDocLoc.docid)).filter(news_sub_query.c.id == OpioidDocLoc.docid).group_by(OpioidDocLoc.city).all()
            elif location_key == 'county':
                stats_query = db.session.query(OpioidDocLoc.county, func.count(OpioidDocLoc.docid)).filter(news_sub_query.c.id == OpioidDocLoc.docid).group_by(OpioidDocLoc.county).all()
            elif location_key == 'state':
                stats_query = db.session.query(OpioidDocLoc.state, func.count(OpioidDocLoc.docid)).filter(news_sub_query.c.id == OpioidDocLoc.docid).group_by(OpioidDocLoc.state).all()

        else:
            if location_key == 'city':
                stats_query = db.session.query(OpioidDocLoc.city, func.count(OpioidDocLoc.docid)).group_by(OpioidDocLoc.city).all()
            elif location_key == 'county':
                stats_query = db.session.query(OpioidDocLoc.county, func.count(OpioidDocLoc.docid)).group_by(OpioidDocLoc.county).all()
            elif location_key == 'state':
                stats_query = db.session.query(OpioidDocLoc.state, func.count(OpioidDocLoc.docid)).group_by(OpioidDocLoc.state).all()

        results = []

        for row in stats_query:

            obj = {}

            if row[0] == ('none' or None) and nones != ('true' or 'True'):
                continue

            obj['location'] = row[0]
            obj['count'] = row[1]

            results.append(obj)

        return make_response(jsonify(results), 200)

    except Exception as e:

        return make_response(jsonify(e), 500)

@articleloc.route('/api/articleloc/_stats/interval', methods=['GET'])
def get_interval_loc_stats():
    """
    group by each key and return the respective values
    """

    interval = request.args.get('interval')

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    date_type = request.args.get('date_type')

    location_key = request.args.get('location_key')
    topic_key = request.args.get('topic_key')
    keywords = request.args.get('keywords')

    result_set = []

    if interval not in ['day', 'week', 'month']:
        return make_response(jsonify([]), 200)

    if not start_date or not end_date:
        return make_response(jsonify([]), 200)

    if date_type not in ['collect', 'publish']:
        date_type = 'publish'

    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')

    if interval == 'day':

        delta = (end_date - start_date).days

        for i in range(delta + 1):

            new_start = start_date + timedelta(days=i)
            new_start = new_start.strftime('%m/%d/%Y')

            qs_list = []

            if location_key:
                qs_list.append('location_key={0}'.format(location_key))
            if topic_key:
                qs_list.append('topic_key={0}'.format(topic_key))
            if keywords:
                qs_list.append('keywords={0}'.format(keywords))

            if date_type == 'collect':
                qs_list.append('before_collection_date={0}'.format(new_start))
            elif date_type == 'publish':
                qs_list.append('before_publish_date={0}'.format(new_start))

            qs = '&'.join(qs_list)

            result = requests.get('http://localhost:5000/api/articleloc/_stats/count?{0}'.format(qs)).json()

            d = {}
            d['date'] = new_start
            d['data'] = result

            result_set.append(d)

    if interval == 'week':

        delta = (end_date - start_date).days

        for i in range(delta + 1):

            if i % 7 == 0 or i == delta:
                new_start = start_date + timedelta(days=i)
                new_start = new_start.strftime('%m/%d/%Y')

                qs_list = []

                if location_key:
                    qs_list.append('location_key={0}'.format(location_key))
                if topic_key:
                    qs_list.append('topic_key={0}'.format(topic_key))
                if keywords:
                    qs_list.append('keywords={0}'.format(keywords))

                if date_type == 'collect':
                    qs_list.append('before_collection_date={0}'.format(new_start))
                elif date_type == 'publish':
                    qs_list.append('before_publish_date={0}'.format(new_start))

                qs = '&'.join(qs_list)

                result = requests.get('http://localhost:5000/api/articleloc/_stats/count?{0}'.format(qs)).json()

                d = {}
                d['date'] = new_start
                d['data'] = result

                result_set.append(d)

    if interval == 'month':

        delta = (end_date - start_date).days

        for i in range(delta + 1):

            if i % 30 == 0 or i == delta:
                new_start = start_date + timedelta(days=i)
                new_start = new_start.strftime('%m/%d/%Y')

                qs_list = []

                if location_key:
                    qs_list.append('location_key={0}'.format(location_key))
                if topic_key:
                    qs_list.append('topic_key={0}'.format(topic_key))
                if keywords:
                    qs_list.append('keywords={0}'.format(keywords))

                if date_type == 'collect':
                    qs_list.append('before_collection_date={0}'.format(new_start))
                elif date_type == 'publish':
                    qs_list.append('before_publish_date={0}'.format(new_start))

                qs = '&'.join(qs_list)

                result = requests.get('http://localhost:5000/api/articleloc/_stats/count?{0}'.format(qs)).json()

                d = {}
                d['date'] = new_start
                d['data'] = result

                result_set.append(d)

    return make_response(jsonify(result_set), 200)




