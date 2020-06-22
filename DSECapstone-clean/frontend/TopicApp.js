import React from "react"
import ReactDOM from 'react-dom';

import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';

import axios from 'axios';

export default class TopicApp extends React.Component { 

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,

      topics: []
    };

    this.getTopics = this.getTopics.bind(this)
  }

  componentDidMount() {
    this.getTopics();
  }

  getTopics() {
    axios.get("api/topic")
      .then(res => res.data)
      .then(
        (result) => {

          this.setState({
            isLoaded: true,
            topics: result
          });
        },
        (error) => {

          this.setState({
            isLoaded: true,
            error: error
          });
        }
      )
  }

  render() {

    const { error, isLoaded, article, topics } = this.state;

    if (error) {
      return <div>
        Error 
      </div>
    } else if (!isLoaded) {
      return <div>
        Loading ...
      </div>
    }

    return (
      <TableContainer component={Paper}>
        <h2 align="center">Expert Topics</h2>
        <Table aria-label="Topic Label Table">
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell align="center">Topic Name</TableCell>
              <TableCell align="center">Topic Description</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {topics.map((topic) => (
              <TableRow key={topic.id}>
                <TableCell component="th" scope="row">
                  {topic.id}
                </TableCell>
                <TableCell align="right">{topic.label_name}</TableCell>
                <TableCell align="right">{topic.label_desc}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    )
  }
}