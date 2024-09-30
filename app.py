from flask import Flask, render_template, request
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pickle 
import xgboost as xgb
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':

        df = pd.read_csv('static/final_data.csv')
        with open('static/CRM_model.pkl', 'rb') as file:
            xgb_classifier = pickle.load(file)

            # Get form data
            form_data = request.form
            
            # Create a LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=df.values,
                feature_names=df.columns.tolist(),
                class_names=[0,1,2,3],  # Replace with actual class names if there are multiple
                mode='classification'
            )

            # Select an instance to explain
            userId = form_data.get('userId')
            userId = int(userId)
            instance = df.iloc[userId].values
            
            # Explain the instance using predict_proba
            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=xgb_classifier.predict_proba,  # Use predict_proba instead of predict
                num_features=15  # Specify the number of features to be included in the explanation
            )
            result = xgb_classifier.predict(instance.reshape(1, -1))
            

            # Generate a matplotlib figure
            fig = explanation.as_pyplot_figure()

            # Customize the figure size
            fig.set_size_inches(10, 6)

            # Add a title
            plt.title('LIME Explanation for Instance')

            # Display the figure
            plt.savefig('static/lime_explanation.png')  # Save as PNG file



        # Return a response (e.g., a thank you message)
        
        return render_template('submited.html', result=result[0])
    else:
        return 'Method Not Allowed', 405

if __name__ == '__main__':
    app.debug = True
    app.run()
