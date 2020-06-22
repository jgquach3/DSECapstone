const path = require('path');

const config = {
  entry: './index.js',
  output:{
    path: __dirname,        //added before output.filename
    filename: "bundle.js"
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        loaders: [
          'babel-loader',
        ],
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: [
          'style-loader',
          'css-loader'
        ]
      },
    ],
  }, 
  // devServer: {
  //   proxy: {
  //     '/api': 'http://localhost:5000'
  //   }
  // }
};

module.exports = config;