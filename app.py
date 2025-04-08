from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
CLEANED_FOLDER = 'cleaned'
REPORT_FOLDER = 'reports'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def scale_data(df, method='minmax'):
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clean', methods=['POST'])
def clean():
    file = request.files['file']
    if not file or file.filename == '':
        return "No file uploaded."

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    report = []

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    report.append(f"Removed {before - after} duplicate rows.")

    missing_action = request.form.get('missing_action')
    if missing_action == 'drop':
        df.dropna(inplace=True)
        report.append("Dropped rows with missing values.")
    elif missing_action == 'mean':
        df.fillna(df.mean(numeric_only=True), inplace=True)
        report.append("Filled missing values with column means.")
    else:
        df.fillna(method='ffill', inplace=True)
        report.append("Forward filled missing values.")

    outlier_checkbox = request.form.get('remove_outliers')
    if outlier_checkbox:
        before = len(df)
        df = remove_outliers(df)
        after = len(df)
        report.append(f"Removed {before - after} outlier rows.")

    scaling_method = request.form.get('scaling')
    if scaling_method:
        df = scale_data(df, method=scaling_method)
        report.append(f"Applied {scaling_method} scaling to numeric columns.")

    cleaned_filename = 'cleaned_' + filename
    cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_filename)
    df.to_csv(cleaned_path, index=False)

    report_filename = 'report_' + filename.replace('.csv', '.txt')
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    return render_template('cleaned.html', filename=cleaned_filename, report=report_filename)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(CLEANED_FOLDER, filename), as_attachment=True)

@app.route('/report/<filename>')
def download_report(filename):
    return send_file(os.path.join(REPORT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
