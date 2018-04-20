from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from model import Model
import csv
import pandas as pd

blueprint = Blueprint(
    'home_blueprint',
    __name__,
    url_prefix='/home',
    template_folder='templates',
    static_folder='static'
)

model = Model()

def process_input_csv(file):
    filename = secure_filename(file.filename)
    reader = pd.read_csv(file)
    reader.fillna(0.0, inplace=True)
    data = reader.drop('ID', axis=1)

    totals = data.sum(axis=0).to_dict().items()
    totals.sort(key=lambda entry: -entry[1])
    data_vals = data.values.tolist()
    predictions = list(model.predict_hainan(data))

    return {
        "forecast": {
            "xs": list(reader.iloc[:,0].values),
            "ys": predictions
        },
        "data": data_vals,
        "totals": totals[:6],
        "stats": {
            "entries": len(list(reader.iloc[:,0].values)),
            "revenue": round(sum(predictions), 2),
            "input": round(sum([entry[1] for entry in totals]), 2),
            "accuracy": round(model.hainan_accuracy * 100, 2)
        }
    }

@blueprint.route('/index')
def index():
    return render_template('index.html')

@blueprint.route('/<template>')
def route_template(template):
    return render_template(template + '.html')

@blueprint.route('/data-upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        return jsonify(process_input_csv(file))
