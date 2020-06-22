import json
from api.database import db
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.inspection import inspect

from sqlalchemy.ext.declarative import declarative_base

from decimal import Decimal

Base = declarative_base()

class Serializer(object):

    def serialize(self):

        j = {}
        for c in inspect(self).attrs.keys():

            attr = getattr(self, c)

            # manually serialize decimal cases ... 
            if isinstance(attr, Decimal):
                j[c] = float(attr)
            else:
                j[c] = attr

        return j

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]

class NewsArticle(db.Model, Serializer):

    __tablename__ = 'usnewspaper'

    id = db.Column('id', db.Integer, primary_key=True)
    news = db.Column('news', db.Text)
    collectiondate = db.Column('collectiondate', db.Date)
    title = db.Column('title', db.Text)
    url = db.Column('url', db.VARCHAR(length=600))
    publishdate = db.Column('publishdate', db.Date)
    author = db.Column('author', ARRAY(db.Text))
    keywords = db.Column('keywords', ARRAY(db.Text))
    src = db.Column('src', db.VARCHAR(length=400))
    language = db.Column('language', db.VARCHAR(length=2))
    newsindex = db.Column('newsindex', db.Integer)

    def __repr__(self):
        return """
            <NewsArticle(id='%s', news='%s', collectiondate='%s', 
            title='%s', url='%s', publishdate='%s', author='%s',
            keywords='%s', src='%s', language='%s', newsindex='%s')>""" % (self.id, self.news, self.collectiondate, 
                self.title, self.url, self.publishdate, self.author, 
                self.keywords, self.src, self.language, self.newsindex)

    def serialize(self):
        d = Serializer.serialize(self)
        return d

# class OpioidDocLoc(db.Model, Serializer):

#     __tablename__ = 'opioiddocloc'

#     id = db.Column('id', db.Integer, primary_key=True)
#     docid = db.Column('docid', db.Integer)
#     locationname = db.Column('locationname', db.VARCHAR(length=100))
#     latitude = db.Column('latitude', db.Integer)
#     longitude = db.Column('longitude', db.Integer)

#     def __repr__(self):
#         return """
#             <OpioidDocLoc(id='%s', docid='%s', locationname='%s',
#             latitude='%s', longitude='%s')>""" % (self.id, self.docid, self.locationname,
#                 self.latitude, self.longitude)

#     def serialize(self):
#         d = Serializer.serialize(self)
#         return d

class OpioidDocLoc(db.Model, Serializer):

    __tablename__ = 'opioiddocloc'

    id = db.Column('id', db.Integer, primary_key=True)
    docid = db.Column('docid', db.Integer)
    city = db.Column('city', db.VARCHAR(length=100))
    county = db.Column('county', db.VARCHAR(length=100))
    state = db.Column('state', db.VARCHAR(length=100))

    def __repr__(self):
        return """
            <OpioidDocLoc(id='%s', docid='%s', city='%s',
            county='%s', state='%s')>""" % (self.id, self.docid, self.city, 
                self.county, self.state)

    def serialize(self):
        d = Serializer.serialize(self)
        return d

class TopicDoc(db.Model, Serializer):

    __tablename__ = 'topicdoc'

    id = db.Column('id', db.Integer, primary_key=True)
    docid = db.Column('docid', db.Integer)
    topic = db.Column('topic', db.Integer)
    modeldate = db.Column('modeldate', db.DateTime)
    prob = db.Column('prob', db.Integer)

    def __repr__(self):
        return """
            <TopicDoc(id='%s', docid='%s', topic='%s', 
            modeldate='%s', prob='%s')>""" % (self.id, self.docid, self.topic, 
                self.modeldate, self.prob)

    def serialize(self):
        d = Serializer.serialize(self)
        return d

class Topic(db.Model, Serializer):

    __tablename__ = 'opioid_label'

    id = db.Column('label_id', db.Integer, primary_key=True)
    label_name = db.Column('label_name', db.VARCHAR(length=600))
    label_desc = db.Column('label_desc', db.VARCHAR(length=600))

    def __repr__(self):

        return """
            <Topic(id='%s', label_name='%s', 
            label_desc='%s')>""" % (self.id, self.label_name, self.label_desc)

    def serialize(self):
        d = Serializer.serialize(self)
        return d

class Topic_Something_Else(db.Model, Serializer):

    __tablename__ = 'opioid_topic_label'

    id = db.Column('id', db.Integer, primary_key=True)
    label_id = db.Column('label_id', db.Integer)
    topic_id = db.Column('topic_id', db.Integer)
    prob = db.Column('prob', db.Integer)

    def __repr__(self):

        return """
            <Topic(id='%s', label_id='%s', 
            topic_id='%s', prob='%s')>""" % (self.id, self.label_id, self.topic_id, self.prob)

    def serialize(self):
        d = Serializer.serialize(self)
        return d


