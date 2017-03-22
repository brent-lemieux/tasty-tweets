

import boto
from boto.s3.key import Key
import os

# Connect to S3 - DO NOT JUST DIRECTLY REPLACE THE BELOW WITH YOUR ACCESS
# AND SECRET ACCESS KEY!! This runs the risk of you pushing them
# up to github. Instead, read them into variables from a .json file or store
# them as environment variables in your .bashrc / .bash_profile and read
# them in from there. See read_aws_credentials.md for how to do this.

import json

def to_bucket():
    with open('../../amazon.json') as key_file:
        keys = json.load(key_file)
        access_key = keys["AWS_ACCESS_KEY_ID"]
        access_secret_key = keys["AWS_SECRET_ACCESS_KEY"]

    file_name = '/../../tweets/{}'.format(os.listdir('/../../tweets')[0])
    bucket_name = "tasty-tweets"
    fil = open(file_name)
    conn = boto.connect_s3(access_key,access_secret_key)
    bucket = conn.get_bucket(bucket_name)
    #Get the Key object of the bucket
    k = Key(bucket)
    #Crete a new key with id as the name of the file
    k.key = file_name
    #Upload the file
    result = k.set_contents_from_file(fil)
    #result contains the size of the file uploaded

    os.remove(file_name)
