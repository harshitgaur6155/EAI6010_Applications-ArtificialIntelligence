from fastai.text.all import *
from flask import Flask, request, Response
import functools
from google.cloud import storage
import json
import os


app = Flask( __name__ )
storage_client = storage.Client( )


@app.route( "/classify", methods = [ "POST" ] ) # when I post to the {base}/classify URL I call this function
def classify_article( ):
    """
    Classifies a news article given its "headline" and "shortDescription".
    """

    #   Load the classifier
    classifier = _load_classifier( )

    #   Classify the input normally
    request_json = request.get_json( )
    headline = request_json.get( "headline", None )
    short_description = request_json.get( "shortDescription", None )
    if headline is not None and short_description is not None:

        #   Make the prediction
        content = f"{headline} \n {short_description}"
        prediction, prediction_possibility, prediction_possibilities = classifier.predict( content )

        #   Build the response
        response_json = { "predictedClass": prediction }
        response_json_string = json.dumps( response_json )
        response = Response( response_json_string,  mimetype='application/json' )
        return response

    #   We are missing content, complain to the user
    else:
        response_json = { "message": "headline and shortDescription are required to make a class prediction" }
        response_json_string = json.dumps( response_json )
        response = Response( response_json_string,  mimetype='application/json' )
        return response


@functools.lru_cache( maxsize = 1 ) # cache and use the first call result
def _load_classifier( ):
    """ Loads the classifier and uses the lru_cache to do it once. """

    #   Download the model from cloud storage to the container’s local disk
    blob = storage_client.bucket( "classifier-model-3" ).get_blob( "news_category_classifier" )
    blob.download_to_filename( "/tmp/model.pkl" )

    #   Load the classifier from disk and return it
    classifier = load_learner( "/tmp/model.pkl" )
    return classifier


if __name__ == "__main__":
    app.run( debug = True, host = "0.0.0.0", port = int( os.environ.get( "PORT", 8080 ) ) )

