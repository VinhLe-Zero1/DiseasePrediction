Steps to run app:

* export FLASK_APP=app.py
* python3 -m flask run

# How to request 
Send a post request with json as follow:
{
    "symptom": "symptomA,symptomB,symptomC"
}

with more than 1 symptom

Reponse: {"result": "sickness"}

Example:

request:

{
    "symptom": "itching"
}

# Note
The name of symptom must be in english, use eng2vi.json or vi2eng.json for reference