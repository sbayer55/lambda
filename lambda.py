"""Handle single image file and generate web output.
"""

import cv2
import urllib
import boto3
import numpy as np
import numpy.linalg as LA

S3 = boto3.client('s3')

def lambda_handler(event, context):
    """Handle on S3 upload
    """
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.unquote_plus(
        event['Records'][0]['s3']['object']['key'].encode('utf8'))
    try:
        response = S3.get_object(Bucket=bucket, Key=key)
        body = response['Body']
        data = body.read()
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 100, 255, 0)
    except Exception as exception:
        print exception
        print 'Error getting object {} from bucket {}. Make sure they exist' \
        'and your bucket is in the same region as this ' \
        'function.'.format(key, bucket)

        raise exception
