# nginx docker file to reverse proxy services

# nginx base image
FROM nginx:latest

# copy nginx configuration file
COPY ./default.conf /etc/nginx/conf.d/

# delete default nginx files
RUN rm -rf /usr.share/nginx/html/*

# expose ports that users can access
EXPOSE 8000

# start nginx daemon
CMD ["nginx", "-g", "daemon off;"]
