//import React from "react";
//import L from "leaflet";
//import "leaflet/dist/leaflet.css";
//import styled from "styled-components"

const pg = require("pg");                         //or ip of 127.0.0.1 instead of ::1
var connectionString = "postgres://postgres:8onApple@localhost:5432/postgres";//postgis_30_sample";
var pgClient = new pg.Client(connectionString);
var cors = require("cors");

//import Leaflet-React ;


//import Map from "./map/map"





pgClient.connect(err =>{
    /*pgClient.query("SELECT * FROM tl_2019_us_zcta510", (err, res)=>{
        res.rows.forEach(row => {
            console.log(row)
        });
    });*/



const express = require('express');
const router = express.Router();
//const path = require('path');
//const connectionString = process.env.DATABASE_URL || 'postgres://localhost:5432/todo';
const path = require('path');

//const app = express()
const port = 3000

//list = require('./request.js').Request; // see  template

  var app = express();
  
app.use(express.static('/Users/adani/Desktop/Leaflet/index.html')); // exposes index.html, per below
app.use(cors());  

app.get("/", function(req, res){
    res.sendFile('/Users/adani/Desktop/Leaflet/index.html')
});

function generalQuery(queryStr){
    var fields;
    console.log(queryStr);
    return pgClient.query(queryStr, (err, res)=>{
        if(err){
            console.log(err)
        }
        else{
            /*res.rows.forEach(row => {
                console.log(row["intptlat10"])
                console.log(row["intptlon10"])
            });*/
            fields = res.rows//.fields.map(field => field.name);
            return res.rows;
        }
        //I still need it to connect
        //pgClient.end()
    });

    //return fields;
}

app.get('/request/:str', function(req, res){
      // run your request.js script
      // when index.html makes the ajax call to www.yoursite.com/request, this runs
      // you can also require your request.js as a module (above) and call on that:
    //res.send(list.getList()); // try res.json() if getList() returns an object or array
    pgClient.query(req.params.str, (err, result)=>{
        if(err){
            console.log(err)
        }
        else{
            /*res.rows.forEach(row => {
                console.log(row["intptlat10"])
                console.log(row["intptlon10"])
            });*/
            fields = result.rows//.fields.map(field => field.name);
            res.send(result.rows);
        }
        //I still need it to connect
        //pgClient.end()
    });
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
});
