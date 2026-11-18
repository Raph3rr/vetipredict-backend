import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_diseases_dataset():
    # Columns: animal_type, age_months, temperature, pulse_rate, respiration_rate,
    # coughing, diarrhea, lethargy, loss_of_appetite, nasal_discharge, lesions, lameness, label
    
    animals = ["Cattle", "Goat", "Sheep", "Pig", "Poultry"]
    
    # Base profiles for healthy animals
    profiles = {
        "Cattle": {"temp": 38.5, "pulse": 60, "resp": 20, "age_range": (6, 120)},
        "Goat": {"temp": 39.0, "pulse": 75, "resp": 18, "age_range": (3, 60)},
        "Sheep": {"temp": 39.0, "pulse": 75, "resp": 18, "age_range": (3, 60)},
        "Pig": {"temp": 39.0, "pulse": 70, "resp": 15, "age_range": (2, 36)},
        "Poultry": {"temp": 41.5, "pulse": 240, "resp": 30, "age_range": (1, 18)}
    }
    
    data = []
    
    # Generate 1500 records
    for _ in range(1500):
        animal = random.choice(animals)
        prof = profiles[animal]
        
        # Decide if healthy or sick (20% healthy, 80% sick with various conditions)
        is_sick = random.random() > 0.20
        
        age = random.randint(*prof["age_range"])
        temp = prof["temp"] + np.random.normal(0, 0.3)
        pulse = prof["pulse"] + np.random.normal(0, 5)
        resp = prof["resp"] + np.random.normal(0, 2)
        
        coughing = 0
        diarrhea = 0
        lethargy = 0
        loss_of_appetite = 0
        nasal_discharge = 0
        lesions = 0
        lameness = 0
        label = "Healthy"
        
        if is_sick:
            if animal == "Cattle":
                disease = random.choice(["Foot and Mouth Disease", "Mastitis", "Anthrax", "Blackleg"])
                if disease == "Foot and Mouth Disease":
                    temp = 40.5 + np.random.normal(0, 0.4)
                    pulse = 80 + np.random.normal(0, 6)
                    resp = 30 + np.random.normal(0, 3)
                    lesions = 1
                    lameness = 1
                    loss_of_appetite = 1
                    label = disease
                elif disease == "Mastitis":
                    temp = 39.6 + np.random.normal(0, 0.3)
                    pulse = 72 + np.random.normal(0, 4)
                    lethargy = 1
                    loss_of_appetite = 1
                    label = disease
                elif disease == "Anthrax":
                    temp = 41.6 + np.random.normal(0, 0.5)
                    pulse = 105 + np.random.normal(0, 8)
                    resp = 45 + np.random.normal(0, 5)
                    lethargy = 1
                    loss_of_appetite = 1
                    nasal_discharge = 1
                    label = disease
                elif disease == "Blackleg":
                    temp = 41.0 + np.random.normal(0, 0.4)
                    pulse = 90 + np.random.normal(0, 6)
                    resp = 35 + np.random.normal(0, 4)
                    lameness = 1
                    lethargy = 1
                    loss_of_appetite = 1
                    label = disease
                    
            elif animal in ["Goat", "Sheep"]:
                disease = random.choice(["Peste des Petits Ruminants (PPR)", "Brucellosis", "Contagious Ecthyma"])
                if disease == "Peste des Petits Ruminants (PPR)":
                    temp = 41.0 + np.random.normal(0, 0.4)
                    pulse = 100 + np.random.normal(0, 6)
                    resp = 35 + np.random.normal(0, 4)
                    coughing = 1
                    diarrhea = 1
                    nasal_discharge = 1
                    loss_of_appetite = 1
                    label = disease
                elif disease == "Brucellosis":
                    temp = 39.8 + np.random.normal(0, 0.3)
                    lethargy = 1
                    label = disease
                elif disease == "Contagious Ecthyma":
                    temp = 39.5 + np.random.normal(0, 0.3)
                    lesions = 1
                    loss_of_appetite = 1
                    label = disease
                    
            elif animal == "Pig":
                disease = random.choice(["Classical Swine Fever", "Erysipelas"])
                if disease == "Classical Swine Fever":
                    temp = 41.2 + np.random.normal(0, 0.4)
                    pulse = 95 + np.random.normal(0, 6)
                    resp = 30 + np.random.normal(0, 4)
                    lethargy = 1
                    diarrhea = 1
                    loss_of_appetite = 1
                    label = disease
                elif disease == "Erysipelas":
                    temp = 40.8 + np.random.normal(0, 0.4)
                    pulse = 85 + np.random.normal(0, 5)
                    lesions = 1
                    lameness = 1
                    label = disease
                    
            elif animal == "Poultry":
                disease = random.choice(["Newcastle Disease", "Coccidiosis", "Avian Influenza"])
                if disease == "Newcastle Disease":
                    temp = 43.5 + np.random.normal(0, 0.5)
                    pulse = 300 + np.random.normal(0, 15)
                    resp = 50 + np.random.normal(0, 5)
                    coughing = 1
                    nasal_discharge = 1
                    diarrhea = 1
                    lethargy = 1
                    label = disease
                elif disease == "Coccidiosis":
                    temp = 42.1 + np.random.normal(0, 0.3)
                    pulse = 260 + np.random.normal(0, 10)
                    resp = 32 + np.random.normal(0, 3)
                    diarrhea = 1
                    lethargy = 1
                    loss_of_appetite = 1
                    label = disease
                elif disease == "Avian Influenza":
                    temp = 44.0 + np.random.normal(0, 0.5)
                    pulse = 310 + np.random.normal(0, 15)
                    resp = 55 + np.random.normal(0, 5)
                    coughing = 1
                    nasal_discharge = 1
                    lethargy = 1
                    loss_of_appetite = 1
                    label = disease
        
        # Round numerical values to make them look realistic
        temp = round(max(35.0, temp), 1)
        pulse = int(max(20, pulse))
        resp = int(max(5, resp))
        
        data.append({
            "animal_type": animal,
            "age_months": age,
            "temperature": temp,
            "pulse_rate": pulse,
            "respiration_rate": resp,
            "coughing": coughing,
            "diarrhea": diarrhea,
            "lethargy": lethargy,
            "loss_of_appetite": loss_of_appetite,
            "nasal_discharge": nasal_discharge,
            "lesions": lesions,
            "lameness": lameness,
            "label": label
        })
        
    df = pd.DataFrame(data)
    df.to_csv("livestock_diseases.csv", index=False)
    print("Generated livestock_diseases.csv with", len(df), "rows.")

def generate_feed_nutrients_dataset():
    feeds = [
        {"food_id": 1, "food_name": "Yellow Maize (Corn)", "serving_g": 1000, "calories_kcal": 3350, "carbs_g": 22, "protein_g": 85, "fat_g": 38, "vitamin_a_mcg": 1500, "vitamin_c_mg": 10, "iron_mg": 2500, "calcium_mg": 200, "zinc_mg": 120},
        {"food_id": 2, "food_name": "Soybean Meal (48%)", "serving_g": 1000, "calories_kcal": 2230, "carbs_g": 60, "protein_g": 480, "fat_g": 15, "vitamin_a_mcg": 100, "vitamin_c_mg": 5, "iron_mg": 6500, "calcium_mg": 3200, "zinc_mg": 550},
        {"food_id": 3, "food_name": "Wheat Bran", "serving_g": 1000, "calories_kcal": 1300, "carbs_g": 120, "protein_g": 150, "fat_g": 40, "vitamin_a_mcg": 20, "vitamin_c_mg": 2, "iron_mg": 11500, "calcium_mg": 1400, "zinc_mg": 180},
        {"food_id": 4, "food_name": "Alfalfa Hay (Late Bud)", "serving_g": 1000, "calories_kcal": 1950, "carbs_g": 280, "protein_g": 180, "fat_g": 20, "vitamin_a_mcg": 45000, "vitamin_c_mg": 150, "iron_mg": 2200, "calcium_mg": 13500, "zinc_mg": 210},
        {"food_id": 5, "food_name": "Fish Meal (60%)", "serving_g": 1000, "calories_kcal": 2900, "carbs_g": 10, "protein_g": 600, "fat_g": 80, "vitamin_a_mcg": 5000, "vitamin_c_mg": 0, "iron_mg": 30000, "calcium_mg": 54000, "zinc_mg": 1200},
        {"food_id": 6, "food_name": "Rice Bran", "serving_g": 1000, "calories_kcal": 2040, "carbs_g": 130, "protein_g": 120, "fat_g": 130, "vitamin_a_mcg": 10, "vitamin_c_mg": 1, "iron_mg": 15000, "calcium_mg": 800, "zinc_mg": 250},
        {"food_id": 7, "food_name": "Cottonseed Meal", "serving_g": 1000, "calories_kcal": 1900, "carbs_g": 140, "protein_g": 410, "fat_g": 20, "vitamin_a_mcg": 150, "vitamin_c_mg": 4, "iron_mg": 10000, "calcium_mg": 1800, "zinc_mg": 320},
        {"food_id": 8, "food_name": "Bone Meal", "serving_g": 1000, "calories_kcal": 0, "carbs_g": 0, "protein_g": 0, "fat_g": 0, "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "iron_mg": 120000, "calcium_mg": 260000, "zinc_mg": 2000},
        {"food_id": 9, "food_name": "Maize Silage", "serving_g": 1000, "calories_kcal": 750, "carbs_g": 230, "protein_g": 25, "fat_g": 8, "vitamin_a_mcg": 12000, "vitamin_c_mg": 80, "iron_mg": 1800, "calcium_mg": 2700, "zinc_mg": 140},
        {"food_id": 10, "food_name": "Limestone (Ground)", "serving_g": 1000, "calories_kcal": 0, "carbs_g": 0, "protein_g": 0, "fat_g": 0, "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "iron_mg": 0, "calcium_mg": 380000, "zinc_mg": 0}
    ]
    
    df = pd.DataFrame(feeds)
    df.to_csv("food_nutrients.csv", index=False)
    print("Generated food_nutrients.csv (feed profiles) with", len(df), "rows.")

if __name__ == "__main__":
    generate_diseases_dataset()
    generate_feed_nutrients_dataset()
