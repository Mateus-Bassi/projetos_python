from projetoTumbrl.tumbrl import app, database
from projetoTumbrl.tumbrl.models import User, Posts

with app.app_context():
    database.create_all()
