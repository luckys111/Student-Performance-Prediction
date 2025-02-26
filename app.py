from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the preprocessor and model
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Handle OneHotEncoder version differences
if hasattr(preprocessor, 'transformers'):
    for name, transformer, columns in preprocessor.transformers:
        if isinstance(transformer, OneHotEncoder):
            if not hasattr(transformer, '_drop_idx_after_grouping'):
                transformer._drop_idx_after_grouping = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        race_ethnicity = request.form['race_ethnicity']
        parental_education = request.form['parental_education']
        lunch = request.form['lunch']
        test_preparation = request.form['test_preparation']
        writing_score = float(request.form['writing_score'])
        reading_score = float(request.form['reading_score'])

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame([[
            gender, race_ethnicity, parental_education, lunch, test_preparation, writing_score, reading_score
        ]], columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'writing_score', 'reading_score'])

        # Preprocess the input data
        input_data_transformed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_transformed)[0]

        # Constrain the prediction to the range [0, 100]
        prediction = np.clip(prediction, 0, 100)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()