
import unittest
import os
import requests
from score import score
import joblib
import subprocess
import time
import pandas as pd
import json
import warnings
import random

warnings.simplefilter("ignore")

class TestScoringFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()
    #     # Load the trained model for testing
    #     cls.loaded_model = joblib.load('sgd_classifier_model.joblib')
        
    #     # Load input texts from the CSV file
    #     cls.test_df = pd.read_csv("test.csv")
    def setUp(self):
        self.model = joblib.load('xgboost_model.pkl')
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')

    def test_score(self):
        text = "example spam text"
        threshold = 0.5
        prediction, propensity = score(text, self.model, threshold)
        self.assertIn(prediction, [True, False])
        self.assertTrue(0 <= propensity <= 1)


    @classmethod
    def tearDownClass(cls) -> None:
        return super().tearDownClass()

    # def test_score(self):
    #     # Load model and vectorizer for testing
    #     model = joblib.load('xgboost_model.joblib')
    #     # Example tests
    #     prediction, propensity = score('example spam text', model, 0.5)
    #     self.assertTrue(isinstance(prediction, bool))
    #     self.assertTrue(0 <= propensity <= 1)

class TestFlaskApp(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.flask_process = subprocess.Popen(["python", "app.py"])
    #     time.sleep(10)  # allow for the flask server to start
    #     cls.test_dataset = pd.read_csv("test.csv")

    def test_flask(self):
        # Start the Flask app
        flask_process = subprocess.Popen(["python", "app.py"])
        time.sleep(10)
        # Test the /score endpoint
        response = requests.post(
            'http://127.0.0.1:5050/score', 
            data=json.dumps({'text': 'example spam text'}), 
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 200)  # Check if the request was successful
        data = response.json()
        self.assertIn('prediction', data)
        self.assertIn('propensity', data)
        # Stop the Flask app
        flask_process.terminate()



class TestDocker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "my-flask-app", "."], check=True)
        
        # Run the Docker container and expose the required port
        cls.container_id = subprocess.check_output(
            ["docker", "run",  "-p", "5050:5050", "my-flask-app"]
        ).decode("utf-8").strip()

        # Load test data from the CSV file
        cls.test_df = pd.read_csv("test_set.csv")
        
        # Wait for the server within the container to start
        time.sleep(5)

    def test_docker(self):
        # Select a random row from the test data
        random_row = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[random_row, 0]

        # Prepare the data for the POST request to the /score endpoint
        test_data = json.dumps({'text': text})
        headers = {"Content-Type": "application/json"}
        response = requests.post('http://127.0.0.1:5050/score', data=test_data, headers=headers)

        # Check that the request was successful
        self.assertEqual(response.status_code, 200)

        # Here you can add more assertions to validate the response content

    @classmethod
    def tearDownClass(cls):
        # Stop and remove the Docker container
        subprocess.run(["docker", "stop", cls.container_id], check=True)
        subprocess.run(["docker", "rm", cls.container_id], check=True)

if __name__ == '__main__':
    unittest.main()
