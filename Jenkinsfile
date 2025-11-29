pipeline {
    agent any

    environment {
        VENV_PATH = "venv"
    }

    stages {

        stage('Environment Setup') {
            steps {
                echo "Setting up Python environment..."
                // Checkout code
                checkout scm
                // Install Python dependencies using Windows commands
                bat 'python -m venv venv'
                bat 'call venv\\Scripts\\pip install --upgrade pip'
                bat 'call venv\\Scripts\\pip install -r requirements.txt'
            }
        }

        stage('Pipeline Run') {
            steps {
                echo "Running MLflow pipeline..."
                bat 'call venv\\Scripts\\python pipeline_runner.py'
            }
        }

        stage('Test Outputs') {
            steps {
                echo "Verifying pipeline outputs..."
                bat 'if exist models\\rf_model.joblib (echo Model exists) else (echo Model missing & exit 1)'
                bat 'if exist models\\metrics.txt (echo Metrics exists) else (echo Metrics missing & exit 1)'
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
