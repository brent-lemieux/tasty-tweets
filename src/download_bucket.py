import boto
import boto.s3.connection

import os
import json
import pickle


try:
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

except:
    import json
    with open('/home/ubuntu/amazon.json') as key_file:
        keys = json.load(key_file)
        access_key = keys["AWS_ACCESS_KEY_ID"]
        secret_key = keys["AWS_SECRET_ACCESS_KEY"]


bucket_name = "tasty-tweets"
conn = boto.connect_s3(access_key, secret_key)
bucket = conn.get_bucket(bucket_name)



tweets = []
for k in bucket.list():
    key = bucket.get_key(k)
    print key.ongoing_restore
    file_name = str(k.name).split('/')[-1]
    try:
        key.get_contents_to_filename(file_name)
        tweets += pickle.load(open(file_name, 'rb'))
        print 'Passed --- ', file_name
    except:
        print 'Failed --- ', file_name
