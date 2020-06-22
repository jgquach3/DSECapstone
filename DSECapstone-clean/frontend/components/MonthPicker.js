import mobiscroll from '@mobiscroll/react-lite';
import '@mobiscroll/react-lite/dist/css/mobiscroll.min.css';


const now = new Date();
const until = new Date(now.getFullYear() + 10, now.getMonth());

export class MonthPicker extends React.Component {
     constructor(props) {
        super(props);

        this.state = {
            val: new Date("2020-12")
        };
    }
    
    render() {
        return (
            <mobiscroll.Form>
                <mobiscroll.FormGroup>
                    <mobiscroll.FormGroupTitle>Add a new credit card</mobiscroll.FormGroupTitle>
                    <label>
                        Name
                        <input type="text" placeholder="Required" />
                    </label>
                    <label>
                        Card
                        <input type="text" placeholder="Credit card number" />
                    </label>
                    <mobiscroll.Date

                        themeVariant="light"
                        dateWheels="mm - MM  yy"
                        dateFormat="mm/yy"
                        min={now}
                        minWidth={100}
                        max={until}
                        value={this.state.val}
                    >
                        <mobiscroll.Input placeholder="Required">Expiration</mobiscroll.Input>
                    </mobiscroll.Date>
                    <label>
                        Security
                        <input type="text" placeholder="3-digit CVV" />
                    </label>
                </mobiscroll.FormGroup>
            </mobiscroll.Form>
        );
    }    
}