import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import recommendation_system as r


app = Flask(__name__)
df_train = r.load_csv()

# Load the trained Logistic Regression model
lr = joblib.load('model/logistic_regression_model.joblib')


job_mapping = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
                   'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
                   'unemployed': 10, 'unknown': 11}

marital_mapping = {'married': 1, 'single': 2, 'divorced': 0}
education_mapping = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
contact_mapping = {'cellular': 0, 'telephone': 1, 'unknown': 2}
month_mapping = {'may': 8, 'jun': 6, 'jul': 5, 'aug': 1, 'oct': 10, 'nov': 9, 'dec': 2, 'jan': 4, 'feb': 3,
                 'mar': 7, 'apr': 0, 'sep': 11}
outcome_mapping = {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3}


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        try:
            users = read_and_convert_csv('data/test_data.csv')
        except pd.errors.EmptyDataError:
            return render_template('index.html', error='Empty file')

        X_new_data = users[['age', 'job', 'marital', 'education', 'defaulter', 'balance', 'housing',
                            'loan', 'contact', 'month', 'duration', 'campaign', 'pdays', 'prev', 'poutcome']]
        predictions = lr.predict(X_new_data)
        users['deposit'] = predictions

        # Reverse mappings for converting numeric values back to categorical
        reverse_job_mapping = {v: k for k, v in job_mapping.items()}
        reverse_marital_mapping = {v: k for k, v in marital_mapping.items()}
        reverse_education_mapping = {v: k for k, v in education_mapping.items()}
        reverse_contact_mapping = {v: k for k, v in contact_mapping.items()}
        reverse_month_mapping = {v: k for k, v in month_mapping.items()}
        reverse_outcome_mapping = {v: k for k, v in outcome_mapping.items()}

        # Apply reverse mappings to convert numeric values back to categorical
        users['job'] = users['job'].map(reverse_job_mapping)
        users['marital'] = users['marital'].map(reverse_marital_mapping)
        users['education'] = users['education'].map(reverse_education_mapping)
        users['contact'] = users['contact'].map(reverse_contact_mapping)
        users['month'] = users['month'].map(reverse_month_mapping)
        users['poutcome'] = users['poutcome'].map(reverse_outcome_mapping)

        users_dict = users.to_dict(orient='records')
        return render_template('index.html', prediction=users_dict)
    return render_template('index.html')


def read_and_convert_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Empty file at '{file_path}'")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV at '{file_path}'")
        return None

    df['job'] = df['job'].map(job_mapping)
    df['marital'] = df['marital'].map(marital_mapping)
    df['education'] = df['education'].map(education_mapping)
    df['contact'] = df['contact'].map(contact_mapping)
    df['month'] = df['month'].map(month_mapping)
    df['poutcome'] = df['poutcome'].map(outcome_mapping)

    df['defaulter'] = df['defaulter'].map({'no': 0, 'yes': 1})
    df['housing'] = df['housing'].map({'no': 0, 'yes': 1})
    df['loan'] = df['loan'].map({'no': 0, 'yes': 1})
    return df


@app.route('/recommendations', methods=['GET'])
def recommendations():
    return render_template('recommendation.html')


@app.route('/product-recommendations', methods=['GET'])
def product_recommendations():
    return render_template('recommendation_user.html')


@app.route('/submit_recommendation', methods=['POST'])
def submit_recommendations():
    data = request.json
    selected_values = data.get('values', [])
    global df_train

    fetcha_dato_value = '2016-05-28'
    user_id = 1061608
    row_16_1061608 = df_train[(df_train['fecha_dato'] == fetcha_dato_value) & (df_train['ncodpers'] == user_id)]
    row_16_1061608.drop(['fecha_dato', 'ncodpers'], axis=1, inplace=True)
    row_15_1061608 = df_train[(df_train['fecha_dato'] == '2015-05-28') & (df_train['ncodpers'] == user_id)]
    row_15_1061608.drop(['fecha_dato', 'ncodpers'], axis=1, inplace=True)

    owned_account_1505 = []
    for column in row_15_1061608.columns:
        if row_15_1061608[column].iloc[0] == 1:
            product_name = r.product_names.get(column, "Unknown")
            owned_account_1505.append(product_name)

    owned_account_1605 = []
    for column in row_16_1061608.columns:
        if row_16_1061608[column].iloc[0] == 1:
            product_name = r.product_names.get(column, "Unknown")
            owned_account_1605.append(product_name)

    df_1505 = get_data_1505()

    df_1505 = r.add_user_input(selected_values, df_1505)
    df_ui = r.df_useritem(df_1505)
    cos_sim = r.cos_sim(df_ui)
    ui = r.useritem(0, df_ui, sim_matrix=cos_sim)
    df_mb = r.df_mb(df_1505)

    hybrid_rec = r.hybrid(0, df_p=df_1505, df_u=df_ui, sim_matrix=cos_sim, df_m=df_mb, f1=0.5, f2=0.25, f3=0.25)

    rec = r.recommendation(0, df_mb, hybrid_rec)
    return jsonify({
        'message': 'Recommendation received successfully!',
        'data': {
            'rec': rec,
            'owned_account_1505': owned_account_1505,
            'owned_account_1605': owned_account_1605
        }
    })


def get_data_1505():
    df_train1505 = r.load_df_1505(df_train)
    return df_train1505


if __name__ == '__main__':
    app.run()
