#!/bin/bash
# init_airflow.sh

# Initialiser la DB
airflow db init

# Créer l'utilisateur Admin
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

