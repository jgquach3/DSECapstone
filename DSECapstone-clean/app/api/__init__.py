from flask import Flask, request, jsonify, make_response, Blueprint
from api.database import db

import csv

def app_factory(alt_config=None):
    """Factory app creation method 

    Args:
        alt_config: Optional Flask config
            if not provided, the production config will be used by default
    
    Returns:
        Flask application
    """

    # import the app logging module here?
    # use a central logger

    app = Flask(__name__, instance_relative_config=True)

    if alt_config is None:
        # use the default production config if separate config is not specified
        app.config.from_pyfile('flask_prod.cfg', silent=True)
    else:
        # if config specified, load that config
        app.config.from_pyfile(alt_config)

    username = app.config['USERNAME']
    password = app.config['PASSWORD']
    hostname = app.config['HOSTNAME']

    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://{0}:{1}@{2}:5432/postgres'.format(username, password, hostname)

    db.init_app(app)

    with app.app_context():
        from api.healthz import healthz
        from api.newsarticle import newsarticle
        from api.topicdoc import topicdoc
        from api.topic import topic
        from api.articleloc import articleloc

        app.register_blueprint(healthz)
        app.register_blueprint(newsarticle)
        app.register_blueprint(topicdoc)
        app.register_blueprint(topic)
        app.register_blueprint(articleloc)
        
    return app
