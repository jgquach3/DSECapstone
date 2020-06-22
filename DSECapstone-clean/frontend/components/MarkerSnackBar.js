import React from 'react';

import Button from '@material-ui/core/Button';
import Snackbar from '@material-ui/core/Snackbar';
import IconButton from '@material-ui/core/IconButton';
import CloseIcon from '@material-ui/icons/Close';

// should rename this later
export default class ArticleSnackBar extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      visible: false,
      messaage: ''
    };

    this.openSnack = this.openSnack.bind(this);
    this.closeSnack = this.closeSnack.bind(this);
  }

  closeSnack(event, reason) {
    if ( reason == 'clickaway' ) {
      return
    }

    this.setState({visible: false})
  }

  openSnack() {
    this.setState({visible: true});
  }

  updateMessageState(val) {
    this.setState({message: val});
  }

  render() {
    return (
      <Snackbar
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        open={this.state.visible}
        autoHideDuration={5000}
        onClose={this.closeSnack}
        message={this.state.message}
        action={
          <React.Fragment>
            <Button color="secondary" size="small" onClick={this.closeSnack}>
              Dismiss
            </Button>
            <IconButton size="small" aria-label="close" color="inherit" onClick={this.closeSnack}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </React.Fragment>
        }
      />
    )
  }

}