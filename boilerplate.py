### Important ###

# This boilerplate code has been provided to help you get started with 
# querying LexisNexis. This script assumes you have already already saved
# your client id and client secret as environment variables.

# It is strongly recommended that you read through all notes and
# comments in this boilerplate script to gain an understanding of how
# it works.

# You are welcome to reuse, rewrite, and reformat this boilerplate code
# into your scripts and to add functionality, such as parsing and saving
# results to a file.

### Environment Variables ###

# This project uses the dontenv Python package.
# https://pypi.org/project/python-dotenv/
# pip install python-dotenv
# add a file called: .env
# to the root directory of your project
# IMPORTANT: if you use git, add .env to .gitignore

#### Apply for Credentials ###

# To request access to the LexisNexis API, fill out this form:
# https://guides.library.stanford.edu/lexis-nexis-api/accessr

import os
import json
import time
import requests
import urllib.parse as urlparse
from urllib.parse import urlencode
from requests.auth import HTTPBasicAuth

import dotenv
from dotenv import load_dotenv

load_dotenv()  # load .env for environment variables

### Parameters to Update ###

# GET Request query string from LexisNexis WSAPI
wsapi_url = "https://services-api.lexisnexis.com/v1/News?$expand=PostFilters,Source&$search=housing%2bvouchers&$select=&$filter=People+eq+%27UEUwMDA5VEpL%27"

#return full text for each search result? True or False
fullText = False

# number of results returned, max 50
top = 1

# number to skip ahead in the results
skip = 0

### Do NOT edit below this line unless you are comfortable with Python ###

# This function queries the LexisNexis API using a GET Request string from the
# LexisNexis Web Services API Interface.
# inputs: wsapi_url - GET Request query string from LexisNexis WSAPI
#         token - from the LexisNexis authentication API (get_token.py)
#         fullText - return full text for each search result? True or False
#         top - number of results returned, max 50
#         skip - number to skip ahead in the results
# outputs: json response from LexisNexisAPI
def query(wsapi_url, token, fullText, top, skip):

    headers = {
                'Authorization' : 'Bearer '+token,
                }

    wsapi_url = update_url(wsapi_url, top, skip, fullText)
    
    if wsapi_url == None:
        return

    print (wsapi_url)
    # send request to LexisNexis API
    content = requests.get(wsapi_url, headers=headers)

    # process results as json
    results = content.json()

    return results

# This function adds top and skip parameters into the URL, as well as modifying
# it to access fullText if that is requested by the user.
# inputs: wsapi_url - GET Request query string from LexisNexis WSAPI
#         top - number of results returned, max 50
#         skip - number to skip ahead in the results
#         fullText - return full text for each search result? True or False
# output: an updated URL for the LexisNexis API
def update_url(wsapi_url, top, skip, fullText):
    url_parts = list(urlparse.urlparse(wsapi_url))
    params = {}

    if type(top) == int:
        params["$top"] = top
    else:
        print ("top must be a number")
        return

    if type(skip) == int:
        params["$skip"] = skip
    else:
        print ("skip must be a number")
        return

    if fullText == True:
        url_parts[2] = "/v1/BatchNews"
        params["$expand"] = "Document"

    if fullText == False:
        try:
            del params["$expand"]
        except:
            pass

    url_query = dict(urlparse.parse_qsl(url_parts[4]))
    url_query.update(params)
    url_parts[4] = urlencode(url_query)
    wsapi_url = urlparse.urlunparse(url_parts)

    return wsapi_url

# This function generates a new LexisNexis API access token.
# inputs: clientid - retrieved from environment variable
#         clientsecret - retrieved from environment variable
# output: an API access token, valid for 24 hours, that is saved to the .env
#         file for access later, along with the time stamp when it expires
def get_token(clientid, clientsecret):

    basic = HTTPBasicAuth(clientid, clientsecret)
    
    url = "https://auth-api.lexisnexis.com/oauth/v2/token"
    data = "grant_type=client_credentials&scope=http%3a%2f%2foauth.lexisnexis.com%2fall"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    # post request including required headers, data, and basic authenticaation
    response = requests.post(url, data=data, headers=headers, auth=HTTPBasicAuth(clientid, clientsecret))
    results = response.json()

    # isolate the token from the full response
    token = results["access_token"]

    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)

    dotenv.set_key(dotenv_file, "lntoken", token)
    dotenv.set_key(dotenv_file, "lnexpire", str(int(time.time()) + results['expires_in']))

    return


# This function checks to see if a current LexisNexis API access token exists.
# If not, it executes get_token() to generate a new token.
# inputs: none, but it checks to see if clientid and clientsecret exist in the
#         environment varaibles.
# output: an API access token
def find_token():

    # access credentials from environment variables
    if "clientid" in os.environ and "clientsecret" in os.environ:
        clientid = os.environ["clientid"]
        clientsecret = os.environ["clientsecret"]

        if "lnexpire" in os.environ:
            token_expire = int(os.environ['lnexpire'])

            # if the token expiration date has not been reached, than retrieve the token
            if token_expire < int(time.time()):
                get_token(clientid, clientsecret)

            output_token = os.environ["lntoken"]

        else: # if no token found in environment variables

            get_token(clientid, clientsecret)
            load_dotenv(override=True) # reload .env file to see the newly saved lntoken
            output_token = os.environ["lntoken"]

    else:
        # see this repo's README.md for instructions on how to set environ variables
        output_token = ""
        print("Error: Client ID and Secret not saved to environment Variables.")

    return output_token


token = find_token()
print(token)
if token != "":
    # executes query of LexisNexis API
    results = query(wsapi_url, token, fullText, top, skip)
    print(results)