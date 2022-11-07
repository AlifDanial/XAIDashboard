from unittest.mock import MagicMock
import sys
sys.modules["xgboost"] = MagicMock()

from pathlib import Path
from flask import Flask

import dash
from dash_bootstrap_components.themes import FLATLY, BOOTSTRAP # bootstrap theme
from explainerdashboard import *

from index_layout import index_layout, register_callbacks
from custom import CustomModelTab, CustomPredictionsTab


pkl_dir = Path.cwd() / "pkls"

app = Flask(__name__)

# clas_explainer = ClassifierExplainer.from_file(pkl_dir / "clas_explainer.joblib")
clas_dashboard = ExplainerDashboard(ClassifierExplainer, 
                    title="Classifier Explainer: Predicting survival on the Titanic", 
                    server=app, url_base_pathname="/classifier/", 
                    header_hide_selector=True)

custom_dashboard = ExplainerDashboard(ClassifierExplainer, 
                        [CustomModelTab, CustomPredictionsTab], 
                        title='Titanic Explainer', header_hide_selector=True,
                        bootstrap=FLATLY,
                        server=app,  url_base_pathname="/custom/")

index_app = dash.Dash(
    __name__, 
    server=app, 
    url_base_pathname="/", 
    external_stylesheets=[BOOTSTRAP])