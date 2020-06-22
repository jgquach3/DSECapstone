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

import { connect } from 'react-redux';

import axios from 'axios';

function mapStateToProps(state) {
  return {
    articles: state.articles
  };
}

class InterestedApp extends React.Component { 

  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      visible: false,

      init: this.props.articles,
      articles: [],
      currArticle: null
    };

    this.getArticleData = this.getArticleData.bind(this);
    this.deleteRow = this.deleteRow.bind(this);

    this.handleDetailOpen = this.handleDetailOpen.bind(this);
    this.handleTopicOpen = this.handleTopicOpen.bind(this);
  }

  getArticleData(val) {

    axios.get("api/newsarticle/" + val)
      .then(res => res.data)
      .then(
        (result) => {

          this.setState({
            articles: [ ...this.state.articles, result ]
          });
        }
      )
  }

  deleteRow(val) {

    const { articles } = this.state;

    for(var i = 0; i < articles.length; i++) {
      if(articles[i].id == val) {
        articles.splice(i, 1);
        break;
      }
    }

    this.setState({ articles }, () => {
      this.props.dispatch({ type: 'REMOVE', payload: val });
    });

  }

  handleTopicOpen(id) {

    if (this.topicDialog) {

      this.topicDialog.toggleDialog();

      if (this.topicDialog.state.visible) {
        this.topicDialog.updateItemState(id);

        // this.setState({ currArticle: id });

      }

    }
  }

  handleDetailOpen(id) {

    if (this.detailDialog) {

      this.detailDialog.toggleDialog();

      if (this.detailDialog.state.visible) {
        this.detailDialog.updateItemState(id);

        // this.setState({ currArticle: id });

      }
    }
  }

  componentDidMount() {

    this.state.init.map((value, index) => {
      this.getArticleData(value);
      console.log(value);
    })

  }



  render() {

    const { error, isLoaded, articles } = this.state;
    // needs to be a list 

    return (
      <div>
        <DetailDialog ref={ref => { this.detailDialog = ref; }} />
        <TopicHDialog ref={ref => { this.topicDialog = ref; }} />

        <TableContainer component={Paper}>
          <h2 align="center">Bookmarked Articles</h2>
          <Table aria-label="Topic Label Table">
            <TableHead>
              <TableRow>
                <TableCell></TableCell>
                <TableCell>Article ID</TableCell>
                <TableCell align="center">Article Name</TableCell>
                <TableCell align="center">Article URL</TableCell>
                <TableCell>View Detailed</TableCell>
                <TableCell>Topic History</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {articles.map((article) => (
                <TableRow key={article.id}>
                  <TableCell>
                    <IconButton onClick={() => this.deleteRow(article.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                  <TableCell component="th" scope="row">
                    {article.id}
                  </TableCell>
                  <TableCell align="right">{article.title}</TableCell>
                  <TableCell align="right">{article.url}</TableCell>
                  <TableCell>
                    <IconButton onClick={() => this.handleDetailOpen(article.id)}>
                      <PageviewIcon />
                    </IconButton>
                  </TableCell>
                  <TableCell>
                    <IconButton onClick={() => this.handleTopicOpen(article.id)}>
                      <MenuBookIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </div>
    )
  }
}

export default connect(mapStateToProps)(InterestedApp);
