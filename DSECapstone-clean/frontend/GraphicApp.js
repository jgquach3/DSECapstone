// file used to have the graphic stuff in it
import React from "react"
import ReactDOM from 'react-dom';

import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';

import { IconButton } from '@material-ui/core';
import DeleteIcon from '@material-ui/icons/Delete';
import PageviewIcon from '@material-ui/icons/Pageview';
import MenuBookIcon from '@material-ui/icons/MenuBook';

import DetailDialog from './components/DetailedViewDialog';
import TopicHDialog from './components/TopicHistoryDialog';

import Grid from '@material-ui/core/Grid';

import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';

import { connect } from 'react-redux';

import axios from 'axios';

function mapStateToProps(state) {
  return {
    articles: state.articles
  };
}

class GraphicApp extends React.Component { 

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,

      init: this.props.articles,
      articles: [],
      topics: [],

      loc_set: {},
      loc_data: [],

      topic_set: {},
      topic_data: []
    };

    this.getArticleData = this.getArticleData.bind(this);
    // this.getLocationData = this.getLocationData.bind(this);
    this.getTopicData = this.getTopicData.bind(this);
  }


  getTopicData(val) {
    axios.get("api/topicdoc/" + val)
      .then(res => res.data)
      .then(
        (result) => {

          for (var i = 0; i < result.length; i++){

            if (result[i].topic in this.state.topic_set) {
              for (var j = 0; j < this.state.topic_data.length; j++) {

                if (this.state.topic_data[j].name == result[i].topic) {
                  this.state.topic_data[j].data += 1
                }
              }
            }
            else {
              this.state.topic_set[result[i].topic] = 1;
              let obj = {};

              obj['name'] = result[i].topic;
              obj['data'] = 1

              this.state.topic_data.push(obj); 
            }
          }
          this.setState({
            topics: [ ...this.state.topics, result ]
          });
        }
      )
  }


  getArticleData(val) {

    axios.get("api/newsarticle/" + val)
      .then(res => res.data)
      .then(
        (result) => {

          console.log('///////')
          console.log(this.state.loc_data);
          console.log('end')

          if (result.state) {
            if(result.state in this.state.loc_set) {
              console.log('HEHREHSALKJ;AL')
              console.log(this.state.loc_data)  
              for (var j = 0; j < this.state.loc_data.length; j++) {

                console.log(this.state.loc_data[j])
                if (this.state.loc_data[j].name == result.state) {
                  this.state.loc_data[j].data += 1
                  console.log(this.state.loc_data)
                  console.log('yup')

                }
              }
            }
            else {
              this.state.loc_set[result.state] = 1;
              let obj = {};

              obj['name'] = result.state;
              obj['data'] = 1

              this.state.loc_data.push(obj); 
            }
          }

          if (result.county) {
            if(result.county in this.state.loc_set) {
              for (var j = 0; j < this.state.loc_data.length; j++) {

                if (this.state.loc_data[j].name == result.county) {
                  this.state.loc_data[j].data += 1
                }
              }
            }
            else {
              this.state.loc_set[result.county] = 1;
              let obj = {};

              obj['name'] = result.county;
              obj['data'] = 1

              this.state.loc_data.push(obj); 
            }
          }

          if (result.city) {
            if(result.city in this.state.loc_set) {
              for (var j = 0; j < this.state.loc_data.length; j++) {

                if (this.state.loc_data[j].name == result.city) {
                  this.state.loc_data[j].data += 1
                }
              }
            }
            else {
              this.state.loc_set[result.city] = 1;
              let obj = {};

              obj['name'] = result.city;
              obj['data'] = 1

              this.state.loc_data.push(obj); 
            }
          }


          this.setState({
            articles: [ ...this.state.articles, result ]
          });
        }
      )
  }

  componentDidMount() {

    this.setState({
      isLoaded: true,
      visible: true
    }, () => {
      this.state.init.map((value, index) => {

        console.log('####');
        console.log(value);

        this.getArticleData(value);
        this.getTopicData(value);
      })
    })

  }



  render() {

    const { error, isLoaded, loc_data, topic_data } = this.state;

    if (error) {
      return (
        <div>
          Error
        </div>
      )
    } else if (!isLoaded) {
      return (
        <div>
          Currently loading ...
        </div>
      )
    } else {

      if (loc_data.length == 0 && topic_data.length == 0) {
        return (
          <div>
            Bookmark articles and then you can see some stuff
          </div>
        )
      }

      return (
        <div>
          <p>some chart stuff here</p>
          <Grid>
          <BarChart
            width={1000}
            height={300}
            key={loc_data}
            data={loc_data}
            margin={{
              top: 5, right: 30, left: 20, bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis key={loc_data} domain={[0, 'dataMax']}/>
            <Tooltip />
            <Legend />
            <Bar dataKey="data" fill="#8884d8" />
          </BarChart>
          <p>place holder for graph of topic distribution</p>
          <BarChart
            width={1000}
            height={300}
            key={topic_data}
            data={topic_data}
            margin={{
              top: 5, right: 30, left: 20, bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis key={topic_data} domain={[0, 'dataMax']} />
            <Tooltip />
            <Legend />
            <Bar dataKey="data" fill="#8884d8" />
          </BarChart>
          </Grid>
        </div>
      )
    }
  }
}

export default connect(mapStateToProps)(GraphicApp);

