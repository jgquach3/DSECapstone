import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Modal from '@material-ui/core/Modal';
import Button from '@material-ui/core/Button';

import axios from 'axios';

function rand() {
  return Math.round(Math.random() * 20) - 10;
}

function getModalStyle() {
  const top = 50 + rand();
  const left = 50 + rand();

  return {
    top: `${top}%`,
    left: `${left}%`,
    transform: `translate(-${top}%, -${left}%)`,
  };
}

const styles = {
  paper: {
    display: 'block',
    width: '600px',
    maxWidth: '40%',
    height: '400px',
    maxHeight: '40%',
    position: 'fixed',
    zIndex: '1001',
    left: '50%',
    top: '50%',
    background: 'white',
    boxShadow: '0 0 60px 10px rgba(0, 0, 0, 0.9)'
  },
  overlay: {
    position: 'fixed',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    zIndex: '1000',
    background: 'rgba(0, 0, 0, 0.6)'
  },
  content: {
    top: '40px',
    left: '40px',
    right: '40px',
    bottom: '40px',
    border: '1px solid #ccc',
    background: '#fff',
    overflow: 'auto',
    WebkitOverflowScrolling: 'touch',
    borderRadius: '4px',
    outline: 'none',
    padding: '20px',
    background: 'rgba(255, 255, 255, 1)'
  }
};

class MarkerModalTest extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      error: null,
      isLoaded: false,
      items: [],
      visible: false,
      id: null
    };

    this.toggleModal = this.toggleModal.bind(this)
    this.updateItemState = this.updateItemState.bind(this)
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevState.id != this.state.id) {
      // should redo api request here
      console.log('api request here');
    }
  };

  toggleModal() {
    this.setState({visible: !this.state.visible});
  }

  updateItemState(val) {
    this.setState({id: val});
  }


  // webpack json has been modified for cors
  // should use same proxy later ... 
  componentDidMount() {

    console.log('called here');

    axios.get("/api/healthz")
    // fetch("http://api:5000/healthz")
      .then(res => res.data)
      .then(
        (result) => {

          console.log('here');
          console.log(result);

          this.setState({
            isLoaded: true,
            items: result.items
          });
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {

          console.log('error')
          this.setState({
            isLoaded: true,
            error
          });
        }
      )
  }


  render() {
    const { error, isLoaded, items } = this.state;

    if (error) {
      return <div>Error: {error.message}</div>;
    } else if (!isLoaded) {
      return <div>Loading...</div>;
    } else {
      return (
        <div>
            <Modal
              aria-labelledby="simple-modal-title"
              aria-describedby="simple-modal-description"
              open={this.state.visible}
              onClose={this.toggleModal}
              style={{...styles.paper, ...styles.overlay, ...styles.content}}
            >
              <div>
                <Button> hi </Button>
                <h2>Simple React Modal</h2>
                <MarkerModalTest />
                </div>
            </Modal>
        </div>
      );
    }
  }

}


const SimpleModalWrapped = withStyles(styles)(MarkerModalTest);

export default SimpleModalWrapped

