# Dockerfile for a simple NGINX server to serve the GUI
# Save this as `Dockerfile` inside the `gui/` directory

FROM nginx:stable-alpine

# Copy the HTML file into the default Nginx web root
COPY index.html /usr/share/nginx/html/
# Copy a custom Nginx config to enable WebSockets for MQTT
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]