import React from 'react';

import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';

import Divider from '@material-ui/core/Divider';

import Button from '@material-ui/core/Button';

import axios from 'axios';

// should rename this later

// probably need to do a join to location and show what the location is as well 
export default class DetailDialog extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,
      id: null,
      article: []
    };

    this.toggleDialog = this.toggleDialog.bind(this)
    this.updateItemState = this.updateItemState.bind(this)
    this.getArticle = this.getArticle.bind(this)
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

  componentDidUpdate(prevProps, prevState) {

    if (prevState.id != this.state.id) {
      this.setState({
        error: null,
        isLoaded: false,
        article: []
      })

      this.getArticle();
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

    let dialogText = '';

    let articleId = 'ID: ';
    let articleNews = '';
    let articleCollectionDate = 'Collection Date: ';
    let articleTitle = 'Title: ';
    let articleURL = 'URL: ';
    let articlePublishDate = 'Publish Date: ';
    let articleAuthor = 'Author: ';
    let articleKeywords = 'Keywords: ';
    let articleSrc = 'Article source: ';
    let articleLanguage = 'Language: ';

    if (error) {
      dialogText = "Error: Request to API Failed"

      return (
        <div>
          <Button label="Dialog" onClick={this.toggleDialog} />
          <Dialog
            title="Detailed View Dialog"
            open={this.state.visible}
            onClose={this.toggleDialog}
          >
            <DialogTitle>
            </DialogTitle>
            <Divider variant="middle" />
            <DialogContent>
              <DialogContentText>
                { dialogText }
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={this.toggleDialog} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </div>
      )

    } else if (!isLoaded) {
      dialogText = "Currently Loading ..."

      return (
        <div>
          <Button label="Dialog" onClick={this.toggleDialog} />
          <Dialog
            title="Detailed View Dialog"
            open={this.state.visible}
            onClose={this.toggleDialog}
          >
            <DialogTitle>
            </DialogTitle>
            <Divider variant="middle" />
            <DialogContent>
              <DialogContentText>
                { dialogText }
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={this.toggleDialog} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </div>
      )

    } else {
      articleId = articleId + this.state.article.id;
      articleNews = articleNews + this.state.article.news;
      articleCollectionDate = articleCollectionDate + this.state.article.collectiondate;
      articleTitle = articleTitle + this.state.article.title;
      articleURL = articleURL + this.state.article.url;
      articlePublishDate = articlePublishDate + this.state.article.publishdate;
      articleAuthor = articleAuthor + this.state.article.author;
      articleKeywords = articleKeywords + this.state.article.keywords;
      articleSrc = articleSrc + this.state.article.src;
      articleLanguage = articleLanguage + this.state.article.language;
    }

    // should have some handler that if it errors, the on click
    // interested button does not exist

    // maybe do a heaader tag and then dialog content text or something?
    return (
      <div>
        <Button label="Dialog" onClick={this.toggleDialog} />
        <Dialog
          title="Detailed View Dialog"
          open={this.state.visible}
          onClose={this.toggleDialog}
        >
          <DialogTitle>
            <div>{ articleId }</div>
            <div>{ articleTitle }</div>
          </DialogTitle>
          <Divider variant="middle" />
          <DialogContent>
            <h4>Article Metadata</h4>
            <DialogContentText>{ articleAuthor }</DialogContentText>
            <DialogContentText>{ articleCollectionDate }</DialogContentText>
            <DialogContentText>{ articlePublishDate }</DialogContentText>
            <DialogContentText>{ articleKeywords }</DialogContentText>
            <DialogContentText>{ articleSrc }</DialogContentText>
            <DialogContentText>{ articleURL }</DialogContentText>
            <DialogContentText>{ articleLanguage }</DialogContentText>
          </DialogContent>
          <Divider variant="middle" />
          <DialogContent>
            <h4>Location Metadata</h4>
            <DialogContentText>State: { this.state.article.state }</DialogContentText>
            <DialogContentText>County: { this.state.article.county }</DialogContentText>
            <DialogContentText>City: { this.state.article.city }</DialogContentText>
          </DialogContent>
          <Divider variant="middle" />
          <DialogContent>
            <h4>Article Text</h4>
            <DialogContentText>
              { articleNews }
            </DialogContentText>
          </DialogContent>
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
