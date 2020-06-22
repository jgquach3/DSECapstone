import React from 'react';

import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';

import Divider from '@material-ui/core/Divider';

import Button from '@material-ui/core/Button';

import axios from 'axios';
import StickyHeadTable from './StickyHeadTabls'


// should rename this later
export default class LocationDialogList extends React.Component {

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
    this.renderArticles = this.renderArticles.bind(this)
    this.setState = this.setState.bind(this)
  }


  componentDidUpdate(prevProps, prevState) {

    console.log('checking...');

    if (prevState.id != this.state.id) {
      this.setState({
        error: null,
        isLoaded: false,
        article: []
      })

    }
  };

  toggleDialog() {

    console.log('weird');

    // should i even be changing this?
    // just create a new component that's diff probably

    if (this.state.visible === false) {
      this.state.visible = !this.state.visible;
    } 
    else{
      this.setState({visible: !this.state.visible});
    }
    
    console.log(this.state);
  }

  updateItemState(val) {
    this.setState({id: val});
    console.log('state');
    console.log(this.state);
  }

  renderArticles() {
    return (
      <List component="nav" aria-label="main mailbox folders">    { 
        this.props.articles.map(a => {
        <ListItem button>
          <ListItemText primary={a.properties.ARTICLE_NAME} />
        </ListItem>
        })}
      </List>
    )

  }
  render() {
    const { error, isLoaded, article, topic, topicProb } = this.state;
    const {articles, title, openArticleModal} = this.props
    console.log('what is this props', this.props)
    let dialogArticleId = 'ID: '
    let dialogTitle = ''
    let dialogText = ''
    let dialogDate = 'Publish Date: '

    let dialogTopic = 'Topic: '
    let dialogTopicDate = 'Topic Date: '
    
    return (
      <div>
        <Button label="Dialog" onClick={this.toggleDialog} />
        <Dialog
          title="Article Dialog"
          open={this.state.visible}
          onClose={this.toggleDialog}
        >

          <DialogTitle>
            <div>{title}</div>

          </DialogTitle>
          <Divider variant="middle" />
          <StickyHeadTable articles = {articles} openArticleModal = {openArticleModal} />
          {/* <List  component="nav" aria-label="main mailbox folders">
            {articles.map(a => 
                  <ListItem key={a.properties.ARTICLE_ID} button onClick={() => {
                      openArticleModal(a.properties.ARTICLE_ID)
                    }
                    }>
                    <ListItemText key={a.properties.ARTICLE_ID}  primary={a.properties.ARTICLE_NAME} />
                  </ListItem>

            )}
           </List> */}

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
