#!/bin/bash
source venv/bin/activate
exec gunicorn -b :5000 --access-logfile - --error-logfile - --timeout=500 --workers=1 --threads=2 --worker-class=gthread manage:app
