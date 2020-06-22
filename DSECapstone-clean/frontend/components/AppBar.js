import React from 'react';
import { makeStyles } from '@material-ui/core/styles';

import { Router, Route, Link } from "react-router-dom";

import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';

import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import InboxIcon from '@material-ui/icons/MoveToInbox';
import MailIcon from '@material-ui/icons/Mail';
import BookmarksIcon from '@material-ui/icons/Bookmarks';
import ExploreIcon from '@material-ui/icons/Explore';
import HelpIcon from '@material-ui/icons/Help';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import BarChartIcon from '@material-ui/icons/BarChart';

import MenuIcon from '@material-ui/icons/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import Menu from '@material-ui/core/Menu';

export default class ButtonAppBar extends React.Component {

  constructor(props){
    super(props);

    this.state = {
      menuOpen: false,
      open: false,
      anchorEl: null,
      setAnchorEl: null
    }

    this.handleMenu = this.handleMenu.bind(this)
    this.handleClose = this.handleClose.bind(this)
  }

  handleMenu(evt) {
    this.setState({ 
      open: true,
      setAnchorEl: evt.currentTarget 
    });
  };

  handleClose(evt) {
    this.setState({ open: false });
  };


  render() {

    return (
      <div>
        <AppBar position="static">
          <Toolbar>
            <IconButton edge="start" color="inherit" aria-label="menu" onClick={this.handleMenu}>
              <MenuIcon />
            </IconButton>
            <Typography variant="h6">
              Opioid
            </Typography>
            <Button color="inherit"></Button>
          </Toolbar>
        </AppBar>
        <Drawer
          variant="persistent"
          open={this.state.open}
        >
          <div>
            <IconButton onClick={this.handleClose}>
              <ChevronLeftIcon />
            </IconButton>
          </div>
          <div style={{width: '250px'}}>
            <List>
              <ListItem button component={ Link } to="/">
                <ListItemIcon><ExploreIcon /></ListItemIcon>
                <ListItemText primary={ 'Map' } />
              </ListItem>
              <ListItem button component={ Link } to="/bookmark">
                <ListItemIcon><BookmarksIcon /></ListItemIcon>
                <ListItemText primary={ 'Bookmarks' } />
              </ListItem>
              <ListItem button component={ Link } to="/charts">
                <ListItemIcon><BarChartIcon /></ListItemIcon>
                <ListItemText primary={ 'Charts' } />
              </ListItem>
              <ListItem button component={ Link } to="/topics">
                <ListItemIcon><HelpIcon /></ListItemIcon>
                <ListItemText primary={ 'Topic Information' } />
              </ListItem>
            </List>
            <Divider />
          </div>
        </Drawer>
      </div>
    );
  }
}
