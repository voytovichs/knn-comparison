from flask import Flask
from flask_injector import FlaskInjector

from vectorsindex.endpoints.health import health_blueprint
from vectorsindex.endpoints.vectors import vectors_blueprint
from vectorsindex.impl import bind_index_implementation

# Flask app
app = Flask(__name__)

# Blueprints
app.register_blueprint(vectors_blueprint)
app.register_blueprint(health_blueprint)

# Dependency Injection
FlaskInjector(
    app=app,
    modules=[bind_index_implementation]
)
