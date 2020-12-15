#!/usr/bin/env bash
COMPOSE_DOCKER_CLI_BUILD=1
DOCKER_BUILDKIT=1

echo killing old docker processes
docker-compose down
docker-compose rm -fs

echo building and running docker container daemon
docker-compose up --build -d
