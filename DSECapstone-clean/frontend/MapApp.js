import React from "react"
import ReactDOM from 'react-dom';
import ReactDOMServer from "react-dom/server";

import { Map, Marker, Popup, TileLayer, Polygon } from "react-leaflet";
import { green } from "ansi-colors";
import { DrugAssociatedOutbreaks } from './components/DrugAssociatedOutbreaks'
import { DrugMarkets } from './components/DrugMarkets'
import { DrugNews } from './components/DrugNews'
import { Fda } from './components/Fda'
import { PrescriptionTrends } from './components/PrescriptionTrends'
import { PublicInterestInDrugs } from './components/PublicInterestInDrugs'

import Button from '@material-ui/core/Button';

import Grid from '@material-ui/core/Grid';
import YearMonthPicker from './components/YearMonthPicker'
import DateFnsUtils from '@date-io/date-fns';
import IconButton from '@material-ui/core/IconButton';
import TodayIcon from '@material-ui/icons/Today';
import InsertInvitationIcon from '@material-ui/icons/InsertInvitation';
import CircularProgress from '@material-ui/core/CircularProgress'
import Input from '@material-ui/core/Input'
import TextField from '@material-ui/core/TextField'
import Backdrop from '@material-ui/core/Backdrop';
import FormControl from '@material-ui/core/FormControl';
import InputLabel from '@material-ui/core/InputLabel';
import Select from '@material-ui/core/Select';
import MenuItem from '@material-ui/core/MenuItem';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import StopIcon from '@material-ui/icons/Stop';
import FastRewindIcon from '@material-ui/icons/FastRewind';
import axios from 'axios';


// import MarkerCard from './components/MarkerPopUp';
import ArticleDialog from './components/MarkerDialog';
import ArticleSnackBar from './components/MarkerSnackBar';
import LocationDialogList from './components/LocationDialogList';
import {keyBy} from 'lodash'
import {interpolateYlOrRd} from 'd3-scale-chromatic'

import { connect } from 'react-redux';

function mapStateToProps(state) {
  return {
    articles: state.articles
  };
}

class MapApp extends React.Component {
  constructor(props) {
    super(props);

    this.minOffset = 0;
    this.maxOffset = 20; 
    const year = (new Date()).getFullYear();
    this.years = Array.from(new Array(20),(val, index) => index + year - 10);
    this.months = Array.from(new Array(13), (val, index) => index)
    this.monthDisplay = ['Jan', 'Feb', 'Mar', 'Spr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'All']

    const thisYear = (new Date()).getFullYear();
    
    console.log("mon", this.months)
    this.state = {
      lat: 40.397,
      lng: -95.270,
      zoom: 4,

      ready: false,
      currentArticleBody: '',
      isShowingArticleBody: true,
      thisYear: thisYear,
      selectedYear: thisYear,
      selectedMonth: this.months[12], 

      // dateBegin: new Date("2019/06/01"),
      dateBegin: new Date("2019/02/01"),
      dateEnd: new Date(),
      
      showModal: false, 
      currArticle: null,
      interestedArticles: {},
      isFetchingData: true,
      articlesList: [],
      articleListTitle: '',
      keyword: '',
      topics: [],
      selectedTopic: "",
      timelapseInProgress: false,
      interval: "month",
      locationKey: 'state',
      timeLapseDate: '',
      pausedTimeLapse: false,
      timeLapseIndex: 0
    };
  
    this.toggleStateBool = true
    this.toggleMPolygons = false
    this.toggleCities = false
    this.toggleCounties = false

    this.countByState = {}
    this.countByCounty = {}
    this.countByCity = {}
    this.updateStateColor = this.updateStateColor.bind(this)
    this.updateCountyColor = this.updateCountyColor.bind(this)
    this.updateCityColor = this.updateCityColor.bind(this)
    this.updateShading = this.updateShading.bind(this)

    this.setReady = this.setReady.bind(this)
    this.toggleStates = this.toggleStates.bind(this)
    // this.toggleMilitaryPolygons = this.toggleMilitaryPolygons.bind(this)
    this.toggleUSCities = this.toggleUSCities.bind(this)
    this.toggleUSCounties = this.toggleUSCounties.bind(this)
    this.update = this.update.bind(this)
    this.handleMarkerClick = this.handleMarkerClick.bind(this)
    this.createClusterIcon = this.createClusterIcon.bind(this)
    this.handleToggleShowArticle = this.handleToggleShowArticle.bind(this)
    this.onHandleYearChange = this.onHandleYearChange.bind(this)
    this.onHandleMonthChange = this.onHandleMonthChange.bind(this)
    this.updateFilter = this.updateFilter.bind(this)
    this.onHandleSelectLocation = this.onHandleSelectLocation.bind(this)

    // time related functions using picker
    this.handleBegin = this.handleBegin.bind(this)
    this.handleEnd = this.handleEnd.bind(this)

    // modal functions
    this.handleModalOpen = this.handleModalOpen.bind(this)
    this.handleListModalOpen = this.handleListModalOpen.bind(this)
    this.interestedArticle = this.interestedArticle.bind(this)
    this.removeInterestedArticle = this.removeInterestedArticle.bind(this)

    //timelapse 
    this.startTimeLapse = this.startTimeLapse.bind(this)
    this.stopTimeLapse = this.stopTimeLapse.bind(this)
    this.pauseTimeLapse = this.pauseTimeLapse.bind(this)
    this.runTimeLapse = this.runTimeLapse.bind(this)
    this.updateColor = this.updateColor.bind(this)
    this.updateOneFrame = this.updateOneFrame.bind(this)
    this.timeLapseState = []
    this.timeLapseId = null


  }
  componentDidMount() {
    
    
    this.map = this.mapInstance.leafletElement
    this.worker = new Worker('worker.js');
    this.worker.postMessage({
      bootup: true,
      filter: {dateBegin: this.state.dateBegin.toLocaleDateString("en-US"), 
        dateEnd: this.state.dateEnd.toLocaleDateString("en-US"), keyword: this.state.keyword, 
        selectedTopic: this.state.selectedTopic
      }
    });

    this.markers_layer = L.geoJson(null, {
        pointToLayer: this.createClusterIcon
    }).addTo(this.map);


    /*

      // set on click handler to open modal or something when opening this
      L.marker(latlng, {icon: icon2, title: i}).on('click', this.handleModalOpen).addTo(this.map);
    }
    */


    this.markers_layer.on('click', (e) => {
      const properties = e.sourceTarget.feature.properties
      if (properties.CONTENT) {
        this.handleMarkerClick(properties.CONTENT)
      }

    });


    this.worker.onmessage =((e) => {
      console.log('what does e get called', e)
      let onHandleSelectLocation = this.onHandleSelectLocation
      let setState = this.setState.bind(this)
        if (e.data.ready) {
            this.setState({topics: e.data.topics})
            this.countByState  = keyBy(e.data.stateStats, 'location')
            this.countByCounty  = keyBy(e.data.countyStats, 'location')
            this.countByCity  = keyBy(e.data.cityStats, 'location')

            let that = this

            this.setReady(true);
            if(!this.state_layer){
                function whenClicked(e) {
                  const name = e.sourceTarget.feature.properties.NAME
                  console.log("click state", e.sourceTarget.feature, e);
                  setState({articleListTitle: `Articles For State: ${name}` })
                  onHandleSelectLocation('state', name)
                }
    
              function onEachFeature(feature, layer) {
                  //bind click
                  layer.on({
                      click: whenClicked
                  });
              }

                this.state_layer = L.geoJson(e.data.polygon_json, {
                    // will do if statement based off how many articles a city has
                    style: that.updateStateColor,
                    onEachFeature: onEachFeature
                }).addTo(this.map)
            }
        if(!this.city_layer){
           function whenClicked(e) {
              const {stateTC, NAME} = e.sourceTarget.feature.properties
              console.log("click city", e.sourceTarget.feature, e);
              setState({articleListTitle: `Articles For: ${NAME}, ${stateTC}` })
              onHandleSelectLocation('city', `${NAME},${stateTC}`)
            }

          function onEachFeature(feature, layer) {
              //bind click
              layer.on({
                  click: whenClicked
              });
          }

          this.city_layer = L.geoJson(e.data.city_json, {
            style: that.updateCityColor,
              onEachFeature: onEachFeature
          })//.addTo(this.map)
        }
        if(!this.county_layer){
          function whenClicked(e) {
             // e = event
             const {stateTC, NAME} = e.sourceTarget.feature.properties
             console.log("click county",e.sourceTarget.feature, e);
             setState({articleListTitle: `Articles For: ${NAME}, ${stateTC}` })
             onHandleSelectLocation('county', `${NAME},${stateTC}`)
           }

         function onEachFeature(feature, layer) {
             //bind click
             layer.on({
                 click: whenClicked
             });
         }


         this.county_layer = L.geoJson(e.data.counties_json, {
            style: that.updateCountyColor,
             onEachFeature: onEachFeature
         })//.addTo(this.map)
        }
            this.update();
        } else if (e.data.expansionZoom) {
            this.map.flyTo(e.data.center, e.data.expansionZoom);
        } else if (e.data.updateFiltered) {
          this.setState({isFetchingData: false})
          this.countByState  = keyBy(e.data.stateStats, 'location')
          this.countByCounty  = keyBy(e.data.countyStats, 'location')
          this.countByCity  = keyBy(e.data.cityStats, 'location')
          console.log('here what is count', this.countByState)
          this.updateShading()
        }else if (e.data.locationNewsDataFetched) {
          this.setState({isFetchingData: false , articlesList: e.data.data})
          this.handleListModalOpen()
        } else if (e.data.startTimeLapse) {
          // BEGIN TIMELAPSE  ==========================================================================================
          this.timeLapseState = e.data.stats
          this.runTimeLapse()
          // END TIMELAPSE  ==========================================================================================
        }
        
        else {
            // this.markers_layer.clearLayers();
            // this.markers_layer.addData(e.data);
        }
        this.setState({isFetchingData: false})
    });
  }
  runTimeLapse() { 
    const that = this
    this.timeLapseId = setInterval(() => { 
      const currIndex = that.state.timeLapseIndex
      console.log("In interval call", that.state.timeLapseIndex)
      if (currIndex >= that.timeLapseState.length) {
        console.log("out of bounds, return")
        that.stopTimeLapse()
        return
      }
      that.updateOneFrame()
      that.setState({timeLapseIndex: that.state.timeLapseIndex+1})
      
    }, 1000)
  }
  updateOneFrame() {
    console.log("what is date",this.state.timeLapseDate, this.timeLapseState[this.state.timeLapseIndex]['date'])
    const layer = `${this.state.locationKey}_layer`
    this.setState({timeLapseDate: this.timeLapseState[this.state.timeLapseIndex]['date']}, () => {
      this[layer].setStyle(this.updateColor)
    })
  }
  updateColor(feature) {
    const article = keyBy( this.timeLapseState[this.state.timeLapseIndex]['data'], 'location')
    const articleVal = article[feature.properties.NAME] ? article[feature.properties.NAME].norm : 0
    const scale = interpolateYlOrRd(articleVal) 
    // console.log("what is color", scale, feature.properties.NAME, this.timeLapseState[this.timeLapseIndex])
    // console.log("selected",  articleVal)
    return {fillColor: `${scale}`, color: 'black',opacity: 0.5, weight: .5}
  }

  updateStateColor(feature) {
    const articleVal = this.countByState[feature.properties.NAME] ? this.countByState[feature.properties.NAME].norm : 0
    const scale = interpolateYlOrRd(articleVal) 
    return {fillColor: `${scale}`, color: 'black',opacity: 0.5, weight: .5}
  }

  updateCountyColor(feature) {
      const articleVal = this.countByCounty[feature.properties.NAME] ? this.countByCounty[feature.properties.NAME].norm : 0
      const scale = interpolateYlOrRd(articleVal) 
      return {fillColor: `${scale}`, color: 'black',opacity: 0.5,weight: .5}
  }
  updateCityColor(feature) {
      const articleVal = this.countByCity[feature.properties.NAME] ? this.countByCity[feature.properties.NAME].norm : 0
      const scale = interpolateYlOrRd(articleVal) 
      return { fillColor: `${scale}`, color: 'black', opacity: 0.5, weight: .5, color: 'black',}
  }
  updateShading() {
      this.city_layer.setStyle(this.updateCityColor)

      this.county_layer.setStyle(this.updateCountyColor)
      
      this.state_layer.setStyle(this.updateStateColor)
  }
  handleMarkerClick(content) {
    this.setState({currentArticleBody: content})
  }
  handleToggleShowArticle() {
    this.setState({isShowingArticleBody: !this.state.isShowingArticleBody})
  }

  createClusterIcon(feature, latlng) {

    if (!feature.properties.cluster){ 
        //zipcode!
        const icon2 = L.divIcon({
            html: `<div><span> <i class="fa fa-newspaper-o"></i></span></div>`,
            className: `marker-cluster marker-cluster-blue`,
            iconSize: L.point(40, 40)
        });

        return L.marker(latlng, {icon: icon2, title: feature.properties.ARTICLE_ID}).on('click', this.handleModalOpen).addTo(this.map);
    }
    
    const count = feature.properties.point_count;
    const size =
        count < 100 ? 'small' :
        count < 1000 ? 'medium' : 'large';

    const icon = L.divIcon({
        html: `<div id="icon-${feature.properties.ARTICLE_ID}"><span>Click For ${ feature.properties.point_count_abbreviated  } More</span></div>`,
        className: `marker-cluster-more marker-cluster-${  size}`,
        iconSize: L.point(60, 60)
    });
    this.setState({isFetchingData: false})
    // .addEventListener('click', () => this.handleMarkerClick(feature.properties.CONTENT))
    return L.marker(latlng, {icon});
  }

  setReady(b){
      const cur = this.state
      cur.ready = b
      this.setState(cur)
  }

  toggleStates(){
    if(this.state_layer){
        if(this.toggleStateBool){
            this.map.removeLayer(this.state_layer)
        }
        else{
            this.map.addLayer(this.state_layer)
        }
        this.toggleStateBool = !this.toggleStateBool


        this.map.removeLayer(this.county_layer)
        this.map.removeLayer(this.city_layer)
        this.toggleCities = false
        this.toggleCounties = false
        this.setState({locationKey: 'state'})
    }

    this.props.dispatch({ type: 'ADD', payload: 4070435 });
    this.props.dispatch({ type: 'ADD', payload: 1 });
    this.props.dispatch({ type: 'ADD', payload: 2925852 });

  }

  toggleUSCities(){
    if(this.city_layer){
      if(this.toggleCities){
          this.map.removeLayer(this.city_layer)
      }
      else{
          this.map.addLayer(this.city_layer)
      }

      this.toggleCities = !this.toggleCities

      this.map.removeLayer(this.state_layer)
      this.map.removeLayer(this.county_layer)
      this.toggleStateBool = false
      this.toggleCounties = false
      this.setState({locationKey: 'city'})
    }
  }


  toggleUSCounties(){
    if(this.county_layer){
      if(this.toggleCounties){
          this.map.removeLayer(this.county_layer)
      }
      else{
          this.map.addLayer(this.county_layer)
      }

      this.toggleCounties = !this.toggleCounties

      this.map.removeLayer(this.state_layer)
      this.map.removeLayer(this.city_layer)
      this.toggleStateBool = false
      this.toggleCities = false
      this.setState({locationKey: 'county'})
    }
  }


  update(){
    if (!this.state.ready) return;
    const bounds = this.map.getBounds();
    

  }
  updateFilter(e) {
    if (e) {
      e.preventDefault();
    }
    const {dateBegin, dateEnd, keyword, selectedTopic} = this.state
    this.setState({isFetchingData: true})
    this.worker.postMessage({
      updateData: true,
      filter: {dateBegin: dateBegin.toLocaleDateString("en-US"), dateEnd: dateEnd.toLocaleDateString("en-US"), keyword, selectedTopic}
    });
  }
  onHandleSelectLocation(locationKey, locationVal) {
    const {dateBegin, dateEnd, keyword, selectedTopic} = this.state
    this.setState({isFetchingData: true})
    this.worker.postMessage({
      updateLocationData: true,
      filter: {
        dateBegin: dateBegin.toLocaleDateString("en-US"), 
        dateEnd: dateEnd.toLocaleDateString("en-US"),
        locationKey,
        locationVal,
        keyword,
        selectedTopic
      }
    });
  }
  onHandleYearChange(evt) {
    // Handle Change Here
    this.setState({ selectedYear: evt.target.value });
  };
  onHandleMonthChange(evt) {
    // Handle Change Here
    this.setState({ selectedMonth: evt.target.value });

    console.log(this.state);
  };

  handleBegin(evt) {
    this.setState({ dateBegin: evt }, ()=> {
      this.updateFilter()
    });

  }

  handleEnd(evt) {
    this.setState({ dateEnd: evt }, () => {
      this.updateFilter()
    });
  }


  handleModalOpen(id) {
    if (this.markerDialog) {
      this.markerDialog.toggleDialog();

      if (this.markerDialog.state.visible) {
        this.markerDialog.updateItemState(id);
        this.setState({ currArticle: id });
      }
    }
  }
  handleListModalOpen(evt) {
    if (this.locationDialogList) {
      this.locationDialogList.toggleDialog();
      console.log("what is evt", evt)
      if (this.locationDialogList.state.visible) {
        // this.locationDialogList.updateItemState(evt.target.options.title);
        // this.setState({ currListArticle: evt.target.options.title });
      }
    }
  }

  interestedArticle() {
    
    // has to set some sort of item here to the key to indicate that there is an interest here
    // should it be the actual contents from the request? (probably right?)

    // could just do a check to see if the key is in the state ,if not we won't the thing at all anyways

    let key = this.state.currArticle;
    
    if (key in this.state.interestedArticles) {
      this.articleSnackBar.updateMessageState('Article: ' + this.state.currArticle + ' has already been added')
    }
    else {
      this.state.interestedArticles[this.state.currArticle] = 'true';
      this.props.dispatch({ type: 'ADD', payload: key });
      this.articleSnackBar.updateMessageState('Article: ' + this.state.currArticle + ' added to Interested')
    }

    this.articleSnackBar.openSnack();

    return
  }

  removeInterestedArticle() {

    // maybe just have a message if the key does not exist
    // to replace the updatestate message with like -> this key does not exist ?

    let key = this.state.currArticle;

    if (key in this.state.interestedArticles) {
      delete this.state.interestedArticles[key];
      this.props.dispatch({ type: 'REMOVE', payload: key });

      this.articleSnackBar.updateMessageState('Article: ' + key + ' removed from Interested')
    }
    else {
      this.articleSnackBar.updateMessageState('Article: ' + key + ' does not exist in Interested')
    }
    
    this.articleSnackBar.openSnack();

    return
  }
  startTimeLapse(e) {
    e.preventDefault();
    this.setState({timelapseInProgress: true})
    const {dateBegin, dateEnd, keyword, selectedTopic, interval, locationKey} = this.state
    this.worker.postMessage({
      startTimeLapse: true,
      filter: {dateBegin: dateBegin.toLocaleDateString("en-US"), dateEnd: dateEnd.toLocaleDateString("en-US"), keyword, selectedTopic, interval, locationKey}
    });
  }
  stopTimeLapse() {
    clearInterval(this.timeLapseId);
    this.setState({timelapseInProgress: false, timeLapseDate: "", timeLapseIndex: 0})
    this.timeLapseState = []
    this.timeLapseId = null
    this.state_layer.setStyle(this.updateStateColor)
  }
  pauseTimeLapse() {
    clearInterval(this.timeLapseId);
    this.setState({pausedTimeLapse: true})
  }


  render() {
    const { thisYear, selectedYear, selectedMonth, isFetchingData, 
      topics, selectedTopic, timelapseInProgress, timeLapseDate, pausedTimeLapse, timeLapseIndex
    } = this.state;
    console.log('what is pauseeeeeddddd', pausedTimeLapse,timeLapseDate )
    const position = [this.state.lat, this.state.lng];
    let timeLapseDateEndUse = "";
    // should the buttons for interest on the dialog be moved outside to the parent?
    function formatDate(date) {
      var d = new Date(date),
          month = '' + (d.getMonth() + 1),
          day = '' + d.getDate(),
          year = d.getFullYear();
  
      if (month.length < 2) 
          month = '0' + month;
      if (day.length < 2) 
          day = '0' + day;
  
      return [month, day, year].join('-');
    }
    if (timeLapseDate) {
      const timeLapseDateDate = new Date(timeLapseDate)
      const timeLapseDateEnd = new Date(timeLapseDateDate.setMonth(timeLapseDateDate.getMonth()+1));

      timeLapseDateEndUse = 'to ' + formatDate(timeLapseDateEnd)
    }

    return (
      <div>
        {isFetchingData && (
          <Backdrop id= 'zind' classes={{ root: 'zind' }}  open={true} onClick={() => {}}>
            <CircularProgress color="inherit" />
          </Backdrop>)
        }
        <Grid container justify="space-around">
          <YearMonthPicker id="begin-picker" label="Beginning of time range" startVal={this.state.dateBegin} handleChangeTo={this.handleBegin}/>
          <YearMonthPicker id="end-picker" label="End of time range" startVal={this.state.dateEnd} handleChangeTo={this.handleEnd}/>
        </Grid>
        <Grid container justify="space-around">
          <form onSubmit={this.updateFilter}>
            <TextField label="Keyword" placeholder="Enter Keyword" onChange={(event) => {
              this.setState({keyword: event.target.value});
            }} />      
            <Button variant="contained" default label="Submit" type="submit" >Update </Button>
          </form>
          <FormControl>
            <InputLabel id="topic-select-label">Topics</InputLabel>
            <Select
              labelId="topic-select-label"
              id="topic-select"
              value={selectedTopic}
              onChange={(event) => {
                this.setState({selectedTopic: event.target.value}, ()=> this.updateFilter(event));
                
              }}
            >
              {topics.map(topic => <MenuItem key={topic.id} value={topic.id}>{topic.label_name}</MenuItem>)}
            </Select>
          </FormControl>
        </Grid>
        <ArticleDialog interestedArticle={ this.interestedArticle } removeInterestedArticle={ this.removeInterestedArticle } ref={ref => { this.markerDialog = ref; }} />
        <ArticleSnackBar  ref={ref => { this.articleSnackBar = ref; }}/>
        <LocationDialogList openArticleModal={this.handleModalOpen} articles={this.state.articlesList} title={this.state.articleListTitle} ref={ref => { this.locationDialogList = ref; }} />

        <Map 
          onMoveend={this.update}
          id="mapid" 
          
          ref={e => { this.mapInstance = e }}

          center={position} 
          zoom={this.state.zoom}>

          <TileLayer
            attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
            url='http://{s}.tile.osm.org/{z}/{x}/{y}.png'
          />
        </Map>
        <Grid container justify="space-around">
        <h5>Timelapse</h5>
        </Grid>
        <Grid container justify="space-around">
          {
            (!timelapseInProgress || pausedTimeLapse) &&    
              (<IconButton aria-label="play" onClick={(e) => {
                  if (pausedTimeLapse) {
                    this.runTimeLapse()
                    this.setState({pausedTimeLapse: false})
                  } else {
                    this.startTimeLapse(e)
                  }
                }
              }>
                <PlayCircleOutlineIcon />
              </IconButton>)
          }
          {(timelapseInProgress && !pausedTimeLapse) && 
              (
              <IconButton aria-label="pause" onClick={this.pauseTimeLapse}>
                <PauseCircleOutlineIcon />
              </IconButton>
              )
          }
          {(timelapseInProgress) && 
              (<IconButton aria-label="stop" onClick={() => {
                if (timeLapseIndex > 0) {
                  
                  this.setState({timeLapseIndex: timeLapseIndex -1 }, () => this.updateOneFrame()) 
                  
                }
              }}>
                <FastRewindIcon />
              </IconButton>
              ) 
          }
          {(timelapseInProgress) && 
              (<IconButton aria-label="stop" onClick={this.stopTimeLapse}>
                <StopIcon />
              </IconButton>
              ) 
          }
        </Grid>
        <Grid container justify="space-around">
        {timelapseInProgress && `Date Range: ${timeLapseDate} ${timeLapseDateEndUse}`}
        </Grid>
        <Grid container justify="space-around">
          <Button variant="contained" color="primary" onClick={this.toggleStates}>Toggle State Boundaries</Button>
          <Button variant="contained" color="primary" onClick={this.toggleUSCounties}>Toggle County Boundaries</Button>
          <Button variant="contained" color="primary" onClick={this.toggleUSCities}>Toggle City Boundaries</Button>
        </Grid>

      </div>
    );
  }
}

// export default MapApp;
export default connect(mapStateToProps)(MapApp);

