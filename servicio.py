from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import re
nltk.download('stopwords')

def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


Model = joblib.load('movie_genres_real.pkl')
tfidfVectorizer = joblib.load('tf_idf_vectorizer.pkl')


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movies genres API',
    description='Movies genres API')

ns = api.namespace('predict', 
     description='Movie Genres Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'text', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


# Definición de la clase para disponibilización
@ns.route('/')
class MoviePrediction(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        data = tfidfVectorizer.transform([args['text']])
        
        return {
         "result": Model.predict_proba(data)
        }, 200
    

# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)  