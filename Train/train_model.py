from sklearn import datasets, svm
import os
from azure.storage.blob import BlockBlobService, PublicAccess

# Get Data
iris = datasets.load_iris()

# Train model
model = svm.LinearSVC()
model.fit(iris.data, iris.target)

# Save model to Blob storage specified in Dockerfile
import pickle
blob_account_name = os.environ.get('ds_blob_account').strip() 
blob_account_key = os.environ.get('ds_blob_key').strip() 
mycontainer = os.environ.get('ds_container').strip()    

filename = os.environ.get('ds_model_filename').strip() 
dirname = os.getcwd()
localfile = os.path.join(dirname, filename)

# save model locally
pickle.dump(model, open(localfile, 'wb'), protocol=2)

# upload model to blob
blob_service=BlockBlobService(account_name=blob_account_name, account_key=blob_account_key)
blob_service.create_blob_from_path(mycontainer, filename, localfile)