import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # Set nama eksperimen
    mlflow.set_experiment("Eksperimen_Produktivitas_Mental")

    mlflow.sklearn.autolog()

    # Load dataset hasil preprocessing
    print("Memuat data preprocessing...")
    data_path = "mental_productivity_preprocessing.csv"
    df = pd.read_csv(data_path)

    # Memisahkan fitur dan target
    X = df.drop(columns=['productivity_score'])
    y = df['productivity_score']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Random_Forest_Basic"):
        print("Memulai pelatihan model Random Forest Regressor...")
        
        model = RandomForestRegressor(random_state=42)
        
        # Autolog akan otomatis merekam proses fit() ini beserta modelnya
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        print(f"Model berhasil dilatih dengan R2 Score pada data validasi: {score:.4f}")

if __name__ == "__main__":
    main()