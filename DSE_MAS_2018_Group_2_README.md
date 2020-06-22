# DSECapstone

repo used for DSE capstone project

clone the repo
cd into DSE Capstone clean 
docker-compose up --build

go to http://localhost:8000 

do a docker ps if you want to see the containers

Commands to rebuild the APP with newest code
`docker-compose up -d --no-deps --build api`
`docker-compose up -d --no-deps --build web`

# Topic modeling & location extraction:

Commands to run the topic modeling:
`cd ./NLP/code; python3 topic_modeling_processing.py`

Commands to run the location extraction:
`cd ./NLP/code; python3 location_extraction.py`



# Sources:

This is where I got the cities geojson
[https://catalog.data.gov/dataset/500-cities-city-boundaries-acd62](https://catalog.data.gov/dataset/500-cities-city-boundaries-acd62)

This is where I got the geojson for state and county
[https://eric.clst.org/tech/usgeojson/](https://eric.clst.org/tech/usgeojson/)

Mapping of state abbreviation
[https://github.com/TexasSwede/stateAbbreviations](https://github.com/TexasSwede/stateAbbreviations)

the geojson simplifier I used to compress the data
[https://mapshaper.org/](https://mapshaper.org/)

The script `transform_LocationBoundaries.js` modified the geojson forms slightly to work better with the map ui.


# Notes:
 requires to have proper connection details within:
 
 `./app/instance/flask_prod.cfg`
 
 `./app/instance/flask_test.cfg`

 they both require 
 USERNAME
 PASSWORD
 HOSTNAME to be filled out.

 This database is hosted by the San Diego Supercomputer Center, and for credentials access, please reach out to Amarnath Gupta.
 
 `./NLP/code/credential.json`
 
 It needs to be in the formatted below: 
 
 ```
 
 {
  "user": "YOUR_USER_NAME_CONNECTING_TO_DATABSE",
  "password" : "YOUR_PASSWORD_CONNECTING_TO_DATABSE"
  }
 
 ```

