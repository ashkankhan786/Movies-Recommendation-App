from flask import Flask , request , jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

df = pd.read_csv('new_database.csv')
df = df.replace({np.nan: None})
notAdult_df = df[df["adult"] == False]


sorted_df = notAdult_df.sort_values(by='popularity', ascending=False)

@app.route('/all', methods=['GET'])
def all():
    sample = sorted_df.head(100)
    sample.replace({np.nan: None}, inplace=True)
    return jsonify(sample.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def predict():
    data = request.get_json()
    desc = data.get('desc')
    year = data.get('year')
    genre = data.get('genre')


    if not desc:
        return jsonify({'error': 'No description provided'}), 400
    
    vectorized_input = vectorizer.transform([desc])
    distances , indices = model.kneighbors(vectorized_input)
    recommendations = df.iloc[indices[0]]
    recommendations = recommendations.sort_values(by='popularity', ascending=False)
    recommendations = recommendations.to_dict(orient='records')



    return jsonify(recommendations)

@app.route("/movie/<movieName>", methods=["GET"])
def get_movie(movieName):
    single_movie = df[df["title"].str.lower() == movieName.lower()].to_dict(orient='records')
    return jsonify(single_movie)


if __name__ == '__main__':
    app.run(debug=False)