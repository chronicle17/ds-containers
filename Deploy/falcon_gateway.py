# falcon_gateway.py
import falcon
import os
import json
from sklearn.externals import joblib
from handle_data import invoke_predict
import pickle
from azure.storage.blob import BlockBlobService, PublicAccess



class InfoResource(object):
    def on_get(self, req, resp):
        """Handles GET Requests"""
        resp.status = falcon.HTTP_200
        resp.body = ('\nThis is an API for to predict flight delays based on weather.\n'
                     'Version: 1.0\n\n'
                     'To learn more, send a GET request to the /predicts endpoint.')


class PredictsResource(object):
    def on_get(self, req, resp):
        """Handles GET Requests"""
        resp.status = falcon.HTTP_200
        resp.body = ('\nRequests and responses served in JSON.\n\n'
                     '<b>Input Schema: <\b>\n'
                     '   month:int\n' 
                     '   day:int\n'
                     '   dayofweek:int\n' 
                     '   departure_hour:int\n'
                     '   windspeed:float\n'
                     '   pressure:float\n'
                     '   precip:float\n'
                     '   dep_airport:str\n'
                     '   arr_airport:str\n'
                     '   carrier:str\n'
                     '<b>Output Schema: <\b>\n'
                     '   delay:int')

    def on_post(self, req, resp):
        """Handles POST Requests"""
        try:
            raw_json = req.stream.read()
        except Exception as ex:
            raise falcon.HTTPError(falcon.HTTP_400,
                                   'Error',
                                   ex.message)

        try:
            result_json = json.loads(raw_json.decode(), encoding='utf-8')
        except ValueError:
            raise falcon.HTTPError(falcon.HTTP_400,
                                   'Malformed JSON',
                                   'Could not decode request body')

        # load model file from Azure
        
        blob_account_name = os.environ.get('ds_blob_account') 
        blob_account_key = os.environ.get('ds_blob_key')  # fill in your blob account key
        mycontainer = os.environ.get('ds_container')     # fill in the container name 

        filename = os.environ.get('ds_model_filename') 
        dirname = os.getcwd()
        localfile = os.path.join(dirname, filename)

        blob_service=BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)
        blob_service.get_blob_to_path(mycontainer, filename, localfile)

        model = pickle.load(open(localfile, 'rb'))

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(invoke_predict(model, raw_json))


app = falcon.API()

info = InfoResource()
predicts = PredictsResource()

app.add_route('/info', info)
app.add_route('/predicts', predicts)
