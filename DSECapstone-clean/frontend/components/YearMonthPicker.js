import React, { useState } from 'react';
import DateFnsUtils from '@date-io/date-fns';
import { MuiPickersUtilsProvider, KeyboardDatePicker } from '@material-ui/pickers';

function YearMonthPicker(props) {
  const [selectedDate, handleDateChange] = useState(props.startVal)


  return (
    <MuiPickersUtilsProvider utils={DateFnsUtils}>
      <KeyboardDatePicker
        margin="normal"
        id={props.id}
        label={props.label}
        format="MM/dd/yyyy"
        value={selectedDate}
        onChange={val => {
          handleDateChange(val);
          props.handleChangeTo(val);
        }}
        KeyboardButtonProps={{
          'aria-label': 'change date',
        }}
      />
    </MuiPickersUtilsProvider>
  );
}

export default YearMonthPicker;
