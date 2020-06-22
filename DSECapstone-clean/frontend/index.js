import React from 'react';
import ReactDOM from 'react-dom';

import MapApp from "./MapApp.js"
import ButtonAppBar from "./components/AppBar.js";
import TopicApp from "./TopicApp.js";
import InterestedApp from './InterestedApp.js';
import GraphicApp from './GraphicApp.js';

import Supercluster from 'supercluster';

import { Container } from '@material-ui/core';

import { BrowserRouter, HashRouter, Switch, Route } from 'react-router-dom';

import { Provider } from 'react-redux';
import { createStore } from 'redux';

// should use a dict or a set instead of an array
// should be unique 
// actually we can just have the command elsewhere?
const initialState = {
  articles: []
};

// these should be a dictionary or something
function reducer(state=initialState, action) {
  console.log('reducer', state, action);

  switch(action.type) {
    case 'ADD':
      return {
        ...state,
        articles: [...state.articles, action.payload]
      };
    case 'REMOVE':
      return {
        ...state,
        articles: state.articles.filter(article => article !== action.payload)
      };
    default:
      return state;
  }

  return state;
}

const store = createStore(reducer);

ReactDOM.render(
  <Provider store={ store }>
    <HashRouter>
      <div>
        <ButtonAppBar />
        <br></br>
        <Container>
          <Switch>
            <Route exact path="/" component={ MapApp } />
            <Route path="/topics" component={ TopicApp } />
            <Route path="/charts" component={ GraphicApp } />
            <Route path="/bookmark" component={ InterestedApp } />
          </Switch>
        </Container>
      </div>
    </HashRouter>
  </Provider>,
  document.querySelector('#root'),
);