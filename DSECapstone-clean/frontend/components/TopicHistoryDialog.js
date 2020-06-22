import React from 'react';

import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';

import { VerticalTimeline, VerticalTimelineElement }  from 'react-vertical-timeline-component';
import 'react-vertical-timeline-component/style.min.css';

import Divider from '@material-ui/core/Divider';

import Button from '@material-ui/core/Button';

import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';

import AccessTimeIcon from '@material-ui/icons/AccessTime';

import axios from 'axios';

// vertial timeline?
// https://www.npmjs.com/package/react-vertical-timeline-component
// or 
// horizontal timeline?
// https://www.npmjs.com/package/react-horizontal-timeline

// leaning towards vertical timeline, which should be cleaner to do overall



// probably need to do a join to location and show what the location is as well 
export default class TopicHDialog extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,
      id: null,
      articles: [], 
      sortedArticles: [],
      sortedKeys: []
    };

    this.toggleDialog = this.toggleDialog.bind(this);
    this.updateItemState = this.updateItemState.bind(this);
    this.getTopicHistory = this.getTopicHistory.bind(this);
    this.sortTopicHistory = this.sortTopicHistory.bind(this);
  }

  sortTopicHistory() {

    let timeKeys = {};
    let sortKeys = [];
    
    for (var i = 0; i < this.state.articles.length; i++) {
      let article = this.state.articles[i];

      if (article.modeldate in timeKeys) {
        timeKeys[article.modeldate].push(article);
      } 
      else {
        timeKeys[article.modeldate] = [article];
      }
    }


    for (var key in timeKeys) {

      sortKeys.push(key)
      let currArticles = timeKeys[key];

      currArticles.sort(function(a, b) {
        var x = a['prob']; var y = b['prob'];
        return ((x > y) ? -1 : ((x < y) ? 1 : 0));
      });

      timeKeys[key] = currArticles;
    }

    sortKeys.sort();
    this.state.sortedKeys = sortKeys;
    this.state.sortedArticles = timeKeys;

    console.log(this.state.sortedKeys);
    console.log(this.state.sortedArticles);

    //  for (var i = 0; i < sortKeys.length; i++) {
    //  let key = sortKeys[i];

    //   for (var j = 0; j < timeKeys[key].length; j++) {
    //     this.state.sortedArticles.push(timeKeys[key][j]);
    //   }
    // }

  }

  getTopicHistory() {

    axios.get("api/topicdoc/" + this.state.id)
      .then(res => res.data)
      .then(
        (result) => {

          this.setState({
            articles: result
          }, ()=> {
            console.log('here')
            this.sortTopicHistory();
            this.setState({ isLoaded: true })
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

  componentDidUpdate(prevProps, prevState) {

    if (prevState.id != this.state.id) {
      this.setState({
        error: null,
        isLoaded: false,
        articles: []
      })

      this.getTopicHistory();
    }
  };

  toggleDialog() {

    if (this.state.visible === false) {
      this.state.visible = !this.state.visible;
    } 
    else{
      this.setState({visible: !this.state.visible});
    }

  }

  updateItemState(val) {
    this.setState({id: val});
  }

  render() {
    const { error, isLoaded, id, sortedKeys } = this.state;

    let dialogText = '';

    if (error) {
      dialogText = "Error: Request to API Failed"

      return (
        <div>
          <Button label="Dialog" onClick={this.toggleDialog} />
          <Dialog
            title="Topic History Dialog"
            open={this.state.visible}
            onClose={this.toggleDialog}
          >
            <DialogTitle>
              <div>hi dog</div>
            </DialogTitle>
            <Divider variant="middle" />
            <DialogContent>
              <DialogContentText>{ dialogText }</DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={this.toggleDialog} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </div>
      );

    } else if (!isLoaded) {
      dialogText = "Currently Loading ..."

      return (
        <div>
          <Button label="Dialog" onClick={this.toggleDialog} />
          <Dialog
            title="Topic History Dialog"
            open={this.state.visible}
            onClose={this.toggleDialog}
          >
            <DialogTitle>
              <div>hi dog</div>
            </DialogTitle>
            <Divider variant="middle" />
            <DialogContent>
              <DialogContentText>{ dialogText }</DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={this.toggleDialog} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </div>
      );

    } else {

      console.log('!!!!!!!');
      console.log(this.state.sortedArticles);
      console.log(sortedKeys)

      return (
        <div>
          <Button label="Dialog" onClick={this.toggleDialog} />
          <Dialog
            title="Topic History Dialog"
            open={this.state.visible}
            onClose={this.toggleDialog}
            fullWidth={true}
            maxWidth = {'md'}
          >
            <DialogTitle>
              <div>Topic History for Article: { id }</div>
            </DialogTitle>
            <Divider variant="middle" />
            <VerticalTimeline>
              {sortedKeys.map(key => {
                return (
                  <VerticalTimelineElement
                    date={ key }
                    layout='1-column'
                    iconStyle={{ background: 'rgb(33, 150, 243)', color: '#fff' }}
                    icon={<AccessTimeIcon />}
                  >
                  <h5>Topic Date: { key }</h5>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell align="center">Topic ID</TableCell>
                        <TableCell align="center">Topic Probability</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {this.state.sortedArticles[key].map(article => {
                        return (
                          <TableRow>
                            <TableCell align="center">{article.topic}</TableCell>
                            <TableCell align="center">{article.prob.toFixed(3)}</TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                  </VerticalTimelineElement>
                )
              })}
            </VerticalTimeline>
            <DialogActions>
              <Button onClick={this.toggleDialog} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </div>
      );
    }
  }
}
