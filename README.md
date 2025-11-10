 🩺 Heart Disease Predictor

 📖 Overview  
The **Heart Disease Predictor** is a machine learning-based web application that predicts the likelihood of a person having heart disease based on medical parameters such as age, cholesterol, blood pressure, fasting blood sugar, and more.  

This project uses a **Logistic Regression** model to make binary classifications (“Likely to have heart disease” or “Not likely”). It also includes a **Streamlit dashboard** that allows users to interactively input their health parameters and view predictions in real time.



 💡 Motivation  
Heart disease remains one of the leading causes of death worldwide. Early detection can help in preventive care. This project demonstrates how **machine learning** can assist in early diagnosis and risk assessment.



 ⚙️ Tech Stack  
- **Python 3.10+**  
- **NumPy** – Numerical computation  
- **Pandas** – Data cleaning and analysis  
- **Scikit-learn** – Machine learning model & preprocessing  
- **Joblib / Pickle** – Saving and loading models  
- **Streamlit** – Interactive user interface (frontend)



 📂 Project Structure  
Heart-Disease-Predictor/
│
├── HeartDiseaseTrain-Test.csv # Dataset
├── model_training.py # ML model training script
├── app.py # Streamlit app for prediction
├── heart_disease_predictor.pkl # Trained model (pickle)
├── scaler.pkl # StandardScaler for input normalization
├── requirements.txt # Python dependencies
└── README.md # Project documentation






🧠 Features  
✅ Reads and preprocesses dataset automatically  
✅ Handles categorical encoding and scaling  
✅ Trains and evaluates a logistic regression model  
✅ Displays accuracy, confusion matrix & classification report  
✅ Saves model and scaler for future use  
✅ Interactive Streamlit UI for real-time predictions  
✅ Clean and reproducible workflow  



 📊 Input Parameters  
The model uses the following key health indicators:

| Feature | Description | Example Values |
|----------|--------------|----------------|
| **age** | Age of the patient | 29–77 |
| **sex** | Gender (1 = male, 0 = female) | 1 |
| **cp** | Chest pain type (0–3) | 2 |
| **trestbps** | Resting blood pressure (mm Hg) | 130 |
| **chol** | Serum cholesterol (mg/dl) | 250 |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | 0 |
| **restecg** | Resting ECG results (0–2) | 1 |
| **thalach** | Maximum heart rate achieved | 150 |
| **exang** | Exercise induced angina (1 = yes; 0 = no) | 0 |
| **oldpeak** | ST depression induced by exercise | 2.3 |
| **slope** | Slope of the ST segment (0–2) | 1 |
| **ca** | Number of major vessels (0–3) | 0 |
| **thal** | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) | 2 |



 🚀 How to Run the Project Locally  

 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/Heart-Disease-Predictor.git
2️⃣ Navigate to the project directory
bash
Copy code
cd Heart-Disease-Predictor
3️⃣ Create a virtual environment
bash
Copy code
python -m venv venv
4️⃣ Activate the virtual environment
Windows:

bash
Copy code
venv\Scripts\activate
Mac/Linux:

bash
Copy code
source venv/bin/activate
5️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
6️⃣ Run the training script
bash
Copy code
python model_training.py
This will:

Load and preprocess the dataset

Train the Logistic Regression model

Evaluate it

Save the model and scaler

🌐 Run the Streamlit App (Frontend Dashboard)
Once your model is trained, run:

bash
Copy code
streamlit run app.py
This will launch the Heart Disease Prediction Dashboard in your browser (default: http://localhost:8501).

🖼️ Example Dashboard Screenshot
Add a screenshot of your Streamlit dashboard here (e.g. dashboard.png)

scss
Copy code
![Heart Disease Dashboard](dashboard.png)
Example sections of the dashboard:
User Input Area: sliders or number inputs for health metrics

Prediction Result: “Likely to have heart disease” / “Not likely”

Accuracy Display: shows model performance metrics

Data Insight Cards: visual stats like average cholesterol or age range

🧾 Example Output (Console)
lua
Copy code
✅ Accuracy: 0.87

Confusion Matrix:
[[24  3]
 [ 5 29]]

Classification Report:
              precision    recall  f1-score   support
           0       0.83      0.89      0.86        27
           1       0.91      0.85      0.88        34
🧩 Model Files Generated
heart_disease_predictor.pkl → Trained Logistic Regression model

scaler.pkl → StandardScaler object

heart_disease_model.pkl → Joblib-saved model

⚠️ Important Notes
Do not upload the venv/ folder to GitHub.

Make sure your .gitignore file contains:

markdown
Copy code
venv/
__pycache__/
*.pkl
Ensure consistent preprocessing (encoding + scaling) during prediction.

Always test your model on unseen data before deployment.

📈 Challenges Faced
Handling missing and non-numeric values

Maintaining consistent preprocessing across training and prediction

Understanding logistic regression behavior on medical data

Managing environment dependencies for smooth Streamlit deployment

📧 Author
👩‍💻 Rajashree Dasgupta
🎓 B.Tech (CSE), Techno India University
📬 Email: dasguptarajo17@gmail.com
💻 GitHub: Rajashree-633

⭐ Acknowledgements
Dataset sourced from the UCI Machine Learning Repository.
Special thanks to open-source contributors and mentors who inspired this project.

💖 Support
If you find this project useful, please ⭐ the repository and share it!
Your support helps me improve and build more ML projects 🚀


