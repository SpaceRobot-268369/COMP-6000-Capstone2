FROM node:20-alpine
WORKDIR /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY package.json ./
RUN npm install
COPY . .
ENTRYPOINT ["/entrypoint.sh"]
CMD ["npm", "run", "dev"]
