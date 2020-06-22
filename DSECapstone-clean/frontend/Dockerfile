# base image
FROM node:12.2.0-alpine

# set working directory
RUN mkdir /app
WORKDIR /app

# copy package and install files
COPY . ./
RUN npm install --silent

EXPOSE 8080 

# run app
CMD ["npm", "start"]