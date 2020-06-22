import React from 'react';

import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';

import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';

import Divider from '@material-ui/core/Divider';

import Button from '@material-ui/core/Button';

import axios from 'axios';

// should rename this later
export default class ArticleDialog extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,
      id: null,
      article: [],

      topic: [],
      topicProb: null
    };

    this.toggleDialog = this.toggleDialog.bind(this)
    this.updateItemState = this.updateItemState.bind(this)
    this.getArticle = this.getArticle.bind(this)
    this.getTopic = this.getTopic.bind(this)
  }

  getArticle() {

    axios.get("api/newsarticle/" + this.state.id)
      .then(res => res.data)
      .then(
        (result) => {

          this.setState({
            isLoaded: true,
            article: result
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

  getTopic() {

    // using the latest flag to 
    axios.get("api/topicdoc/" + this.state.id + "?latest=True")
      .then(res => res.data)
      .then(
        (result) => {

          this.setState({
            topic: result,
            topicProb: result.prob.toFixed(2)
          });
        }
      )
  }

  componentDidUpdate(prevProps, prevState) {

    if (prevState.id != this.state.id) {
      this.setState({
        error: null,
        isLoaded: false,
        article: []
      })

      this.getArticle();
      this.getTopic();
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
    const { error, isLoaded, article, topic, topicProb } = this.state;

    // should include all the sources
    let dialogArticleId = 'ID: '
    let dialogTitle = ''
    let dialogText = ''
    let dialogDate = 'Publish Date: '

    let dialogTopic = 'Topic: '
    let dialogTopicDate = 'Topic Date: '

    if (error) {
      dialogText = "Error: Request to API Failed"
    } else if (!isLoaded) {
      dialogText = "Currently Loading ..."
    } else {
      dialogArticleId = dialogArticleId + this.state.article.id;
      dialogTitle = dialogTitle + this.state.article.title;
      dialogText = this.state.article.news;
      dialogDate = dialogDate + this.state.article.publishdate;

      dialogTopic = dialogTopic + this.state.topic.topic + ' ( ' + this.state.topicProb + ' )' + ' ( ' + this.state.topic.modeldate + ' )';
    }

    return (
      <div>
        <Button label="Dialog" onClick={this.toggleDialog} />
        <Dialog
          title="Article Dialog"
          open={this.state.visible}
          onClose={this.toggleDialog}
        >
          <DialogTitle>
            <div>{ dialogArticleId }</div>
            <div>{ dialogTitle }</div>
            <p>{ dialogDate }</p>
            <p>{ dialogTopic }</p>
          </DialogTitle>
          <Divider variant="middle" />
          <DialogContent>
            <DialogContentText>
              { dialogText }
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={this.props.removeInterestedArticle} color="primary">
              Remove Bookmark
            </Button>
            <Button onClick={this.props.interestedArticle} color="primary">
              Bookmark
            </Button>
            <Button onClick={this.toggleDialog} color="primary">
              Close
            </Button>
          </DialogActions>
        </Dialog>
      </div>
    );
  }
}
