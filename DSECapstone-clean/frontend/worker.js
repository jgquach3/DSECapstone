importScripts('./node_modules/supercluster/dist/supercluster.js');
importScripts('./node_modules/axios/dist/axios.js');

const now = Date.now();

let index;
let geoJsonSaved;
let p_jsonSaved;
let m_polygonsSaved;
let cities_jsonSaved;


async function getGeoJsonNews(filter) {
    const keywordParam = `&keywords=${filter.keyword}`
    const selectedTopicParam = `&topic_key=${filter.selectedTopic}`
    // oldURL = `http://localhost:8000/api/newsarticle?after_publish_date=${e.data.filter.dateBegin}&before_publish_date=${e.data.filter.dateEnd}`
    let newURL = `http://localhost:8000/api/newsarticle?after_publish_date=${filter.dateBegin}&before_publish_date=${filter.dateEnd}&location=True&columns=id,title,publishdate,city,county,state,keywords&location_key=${filter.locationKey}&location_value=${filter.locationVal}${filter.keyword ? keywordParam : ""}${filter.selectedTopic ? selectedTopicParam : ""}`

    const newsData = await axios.get(newURL)
    console.log('what is newsdataa', newsData)
    return newsData.data.map(a => {
        const properties = {
            "ARTICLE_ID": a.id,
            "ARTICLE_NAME": a.title,
            "ARTICLE_DATE": a.publishdate,
            "STATE": a.state,
            "COUNTY": a.county,
            "CITY": a.city,
            "CONTENT": a.contents,
            "KEYWORDS": a.keywords.join(', ')
        }
        return {
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[a.longitude + Math.random() * 2, a.latitude + Math.random() * 2]},
            properties,
            ...properties
        }
    })
}

async function getStats(locationKey, filter) {
  const keywordParam = `&keywords=${filter.keyword}`
  const selectedTopicParam = `&topic_key=${filter.selectedTopic}`
  let url = `http://localhost:8000/api/articleloc/_stats/count?after_publish_date=${filter.dateBegin}&before_publish_date=${filter.dateEnd}&location_key=${locationKey}${filter.keyword ? keywordParam : ""}${filter.selectedTopic ? selectedTopicParam : ""}`
  return axios.get(url)
}
// http://localhost:8000/api/articleloc/_stats/interval?keywords=apple&location_key=county&start_date=1/10/2020&end_date=5/10/2020&interval=month

function LogSlider(options) {
  options = options || {};
  this.minpos = options.minpos || 0;
  this.maxpos = options.maxpos || 100;
  this.minlval = Math.log(options.minval || 1);
  this.maxlval = Math.log(options.maxval || 100000);

  this.scale = (this.maxlval - this.minlval) / (this.maxpos - this.minpos);
}

LogSlider.prototype = {
  // Calculate value from a slider position
  value: function(position) {
     return Math.exp((position - this.minpos) * this.scale + this.minlval);
  },
  // Calculate slider position from a value
  position: function(value) {
     return this.minpos + (Math.log(value) - this.minlval) / this.scale;
  }
};


async function getStatsNormalized(filter) {
  const stateStatsPromise = getStats('state', filter)
  const countyStatsPromise = getStats('county', filter)
  const cityStatsPromise = getStats('city', filter)

  let [stateStats, countyStats, cityStats] = await Promise.all([stateStatsPromise, countyStatsPromise, cityStatsPromise])
  stateStats = stateStats.data
  countyStats = countyStats.data
  cityStats = cityStats.data

  const maxState = Math.max.apply(Math, stateStats.map(function(o) { return o.count; })) 
  const maxCounty = Math.max.apply(Math, countyStats.map(function(o) { return o.count; }))
  const maxCity = Math.max.apply(Math, cityStats.map(function(o) { return o.count; }))

  const minState = Math.min.apply(Math, stateStats.map(function(o) { return o.count; })) 
  const minCounty = Math.min.apply(Math, countyStats.map(function(o) { return o.count; })) 
  const minCity = Math.min.apply(Math, cityStats.map(function(o) { return o.count; }))

  var logslState = new LogSlider({minpos: .0001, maxpos: .9999, minval: minState, maxval: maxState});
  var logslCounty = new LogSlider({minpos: .0001, maxpos: .9999, minval: minCounty, maxval: maxCounty});
  var logslCity = new LogSlider({minpos: .0001, maxpos: .9999, minval: minCity, maxval: maxCity});
  
  const stateStatsNorm = stateStats.map(s => Object.assign({}, s, {norm: logslState.position(s.count)}))
  const countyStatsNorm = countyStats.map(s => Object.assign({}, s, {norm: logslCounty.position(s.count)}))
  const cityStatsNorm = cityStats.map(s => Object.assign({}, s, {norm: logslCity.position(s.count)}))
  console.log('what is statestats norm',minState, maxState,  stateStatsNorm)
  return [stateStatsNorm, countyStatsNorm, cityStatsNorm]
}

async function getIntervalNormalized(filter) {
  const keywordParam = `&keywords=${filter.keyword}`
  const selectedTopicParam = `&topic_key=${filter.selectedTopic}`
  let url = `http://localhost:8000/api/articleloc/_stats/interval?date_type=publish&interval=${filter.interval}&start_date=${filter.dateBegin}&end_date=${filter.dateEnd}&location_key=${filter.locationKey}${filter.keyword ? keywordParam : ""}${filter.selectedTopic ? selectedTopicParam : ""}`

  let intervalStats = await axios.get(url)
  intervalStats = intervalStats.data
  // const maxState = Math.max.apply(Math, stateStats.map(function(o) { return o.count; })) / 2

  // const stateStatsNorm = stateStats.map(s => Object.assign({}, s, {norm: s.count/maxState}))
  console.log('what is interval stats', intervalStats)
  return intervalStats.map(interval => {
    const high = Math.max.apply(Math, interval.data.map(function(o) { return o.count; })) 
    const low = Math.min.apply(Math, interval.data.map(function(o) { return o.count; }))
    var logsl = new LogSlider({minpos: .0001, maxpos: .9999, minval: low, maxval: high});
    const statsNorm = interval.data.map(s => Object.assign({}, s, {norm: logsl.position(s.count)}))
    return {date: interval.date, data: statsNorm}
  })
}

async function getTopics() {
  let url = `http://localhost:8000/api/topic`
  return axios.get(url)
}
//replying to its own message
self.onmessage = async function (e) {
    if (e.data.bootup) {
        const [stateStats, countyStats, cityStats] = await getStatsNormalized(e.data.filter)
        // const geojsonNews = getGeoJsonNews(e.data.filter)
        const topics = await getTopics()
        // console.log("how many pts begin", geojsonNews.length)
        getJSON("data/state_geo_simple.json", "data/FY18_MIRTA_Boundaries.json", "data/city_geo_simple.json", "data/county_geo_simple.json", 
            (state_polygon_json, m_polygons_json, cities_json, counties_json) => {
                console.log("what is geojson", counties_json)

                console.log('here i am?')
            
                // geoJsonSaved = geojsonNews
                // console.log(index.getTile(0, 0, 0));
              console.log('what topic data in worker', topics.data)
                postMessage({
                    ready: true,
                    polygon_json: state_polygon_json,
                    MPolygon_json: m_polygons_json,
                    city_json: cities_json,
                    counties_json,
                    stateStats,
                    countyStats,
                    cityStats,
                    topics: topics.data
                });
        });
    } else {
        if (e.data.getClusterExpansionZoom) {
            // postMessage({
            //     expansionZoom: index.getClusterExpansionZoom(e.data.getClusterExpansionZoom),
            //     center: e.data.center
            // });
        // } else if (e.data.bbox || e.data.zoom) {
        //     postMessage(index.getClusters(e.data.bbox, e.data.zoom));
        } else if (e.data.updateData) { 
            const [stateStats, countyStats, cityStats] = await getStatsNormalized(e.data.filter)
            this.postMessage({updateFiltered: true, stateStats, countyStats, cityStats})
        } else if (e.data.updateLocationData) {
            const geojsonNews = await getGeoJsonNews(e.data.filter)
            console.log("how many pts after select location", geojsonNews.length)
            this.postMessage({locationNewsDataFetched: true, data: geojsonNews})
        } else if (e.data.startTimeLapse) { 
          const stats = await getIntervalNormalized(e.data.filter)
          this.postMessage({startTimeLapse: true, stats})
      } 
    }
};
function makeRequest (method, url) {
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      xhr.open(method, url);
      xhr.setRequestHeader('Accept', 'application/json');
      xhr.responseType = 'json';
      
      
      xhr.onload = function () {
        if (xhr.readyState === 4 && xhr.status >= 200 && xhr.status < 300 && xhr.response) {
          resolve(xhr.response);
        } else {
          reject({
            status: this.status,
            statusText: xhr.statusText
          });
        }
      };
      xhr.onerror = function () {
        reject({
          status: this.status,
          statusText: xhr.statusText
        });
      };
      xhr.send();
    });
  }

async function getJSON( url1, url2, url3, url4, callback) {
    // const stateGeoJson = await makeRequest('GET', url1)
    const militaryGeoJson = [] 
    // const militaryGeoJson = await makeRequest('GET', url2)
    // const cityGeoJson = await makeRequest('GET', url3)
    // const countyGeoJson = await makeRequest('GET', url4)
    const [stateGeoJson, cityGeoJson, countyGeoJson] = await Promise.all([
        makeRequest('GET', url1),
        makeRequest('GET', url3),
        makeRequest('GET', url4)]
    )

    callback(stateGeoJson, militaryGeoJson, cityGeoJson, countyGeoJson);
    return;

}