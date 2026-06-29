# ================================================================
# Frontend Dockerfile
# Stage 1: build with Node 22
# Stage 2: serve with nginx
# ================================================================

FROM node:22-slim AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ================================================================
# Production: serve dist/ with nginx
# ================================================================
FROM nginx:alpine AS production

COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config that handles React Router's client-side routing
# (without this, refreshing any non-root URL returns 404)
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
