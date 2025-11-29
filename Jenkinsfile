pipeline {
    agent any

    environment {
        VENV_PATH = "venv"
    }

    stages {

        // ------------------------
        stage('Environment Setup') {
            steps {
                echo "Setting up Python environment..."
                // Checkout code
                checkout scm
                // Install Python dependencies
                sh 'python -m venv venv'
                sh '${VENV_PATH}/Scripts/pip install --upgrade pip'
                sh '${VENV_PATH}/Scripts/pip install -r requirements.txt'
            }
        }

        // ------------------------
        stage('Pipeline Run') {
            steps {
                echo "Running MLflow pipeline..."
                // Run your MLflow pipeline runner
                sh '${VENV_PATH}/Scripts/python pipeline_runner.py'
            }
        }

        // ------------------------
        stage('Test Outputs') {
            steps {
                echo "Verifying pipeline outputs..."
                // Check if model and metrics were generated
                sh 'test -f models/rf_model.joblib'
                sh 'test -f models/metrics.txt'
            }
        }
    }

    post {
        success {
            echo "MLflow pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed. Check logs."
        }
    }
}
