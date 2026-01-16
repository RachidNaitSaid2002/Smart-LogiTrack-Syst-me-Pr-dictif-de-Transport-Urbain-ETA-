# Smart-LogiTrack: Urban Transport Prediction System

An intelligent predictive system designed to estimate taxi trip arrival times (ETA) in urban environments using machine learning and real-time data processing.

## Overview

Smart-LogiTrack leverages a modern data engineering stack to process NYC Taxi trip data, train predictive models, and serve insights via a RESTful API. The system handles the end-to-end flow from raw data ingestion to real-time ETA predictions.

### Key Features
- **Automated Data Pipeline**: Airflow DAGs for Extract, Load, Transform (ELT) and model training.
- **Machine Learning**: Gradient Boosted Tree (GBT) Regressor trained with PySpark for accurate duration predictions.
- **REST API**: FastAPI backend for real-time predictions and analytics.
- **Authentication**: Secure user Signup/Login with JWT (JSON Web Tokens).
- **Analytics**: Endpoints for exploring traffic patterns (e.g., Average duration by hour).
- **Containerized**: Fully Dockerized environment for easy deployment.

## Architecture

The project is composed of the following microservices orchestrated by Docker Compose:

- **Backend (`backend`)**: FastAPI application serving predictions and analytics.
- **Airflow (`airflow`)**: Orchestrator for the data pipeline:
    1.  **Extract**: Downloads dataset from source.
    2.  **Bronze Layer**: Saves raw data (Parquet).
    3.  **Silver Layer**: Cleans data, performs feature engineering, and loads to PostgreSQL.
    4.  **Training**: Trains GBTRegressor model and saves it.
- **Database (`postgres`)**: PostgreSQL 13 instance for storing app data and Airflow metadata.

## Getting Started

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-
    ```

2.  **Environment Variables**
    Ensure you have a `.env` file in the root directory (refer to `docker-compose.yml` for required variables like `database`, `user`, `password`, `host`).

3.  **Start the Application**
    Initialize and start the containers:
    ```bash
    docker-compose up --build -d
    ```

4.  **Access Services**
    - **Airflow UI**: [http://localhost:8080](http://localhost:8080) (Default credentials: `airflow`/`airflow`)
    - **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

## Usage

### 1. Run Data Pipeline (Airflow)
Before using the API, you need to prepare the data and train the model.
1.  Go to Airflow UI ([http://localhost:8080](http://localhost:8080)).
2.  Trigger the **`ETL_TAXI_TRIP`** DAG.
3.  Wait for all tasks (`Extract`, `Save_Bronze`, `Prepare_Silver`, `Build_Model_Training`) to complete successively.

### 2. Use the API
Once the model is trained, use the Swagger UI ([http://localhost:8000/docs](http://localhost:8000/docs)) to interact with the backend.

- **POST /signup**: Register a new user.
- **POST /login**: Authenticate and get an access token.
- **POST /predictions/**: Submit trip details to get an estimated duration.
    - *Requires Authentication (Bearer Token)*
- **GET /analytics/avg-duration-by-hour/{pickup_hour}**: Get traffic insights.

## Technology Stack
- **Language**: Python 3.8+
- **Web Framework**: FastAPI
- **Data Processing**: Apache Spark (PySpark), Pandas
- **Orchestration**: Apache Airflow
- **Database**: PostgreSQL
- **DevOps**: Docker, Docker Compose

## Project Structure
```
├── airflow/            # Airflow DAGs and configuration
│   └── dags/           # ETL and Training pipelines (ELT.py)
├── backend/            # FastAPI application source code
│   ├── Auth/           # Authentication logic
│   ├── Models/         # SQLAlchemy models
│   ├── Schemas/        # Pydantic schemas
│   └── main.py         # App entry point
├── ml/                 # Machine Learning resources
│   └── NoteBooks/      # Jupyter notebooks for experimentation
└── docker-compose.yml  # Container orchestration
```
