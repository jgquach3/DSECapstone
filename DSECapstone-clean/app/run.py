import api

app = api.app_factory('flask_test.cfg')

if __name__ == '__main__':
    
    app.run(host='0.0.0.0')