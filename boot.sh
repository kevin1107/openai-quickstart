#!/bin/bash
python3 -m venv venv
source venv/bin/activate
exec gunicorn -b :5000 --access-logfile - --error-logfile - manage:app
