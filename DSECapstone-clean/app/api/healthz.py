from flask import Blueprint, jsonify, make_response

healthz = Blueprint('healthz', __name__)

@healthz.route('/api/healthz', methods=['GET'])
def health_check():
    """Health check to ensure that the service is active
    If service is availble, will return a simple "OK" and a 200
    """

    return make_response(jsonify({'message': 'OK'}), 200)