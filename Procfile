web: python railway_server.py
worker: python -m celery worker --app=agents.server:celery_app --loglevel=info
release: python scripts/migrate_railway.py 