from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Literal
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
import joblib
import pickle

course_names = ['AI_Basics', 'Algorithms_Advanced', 'Algorithms_Basics',
       'Cloud_Computing_Advanced', 'Cloud_Computing_Basics',
       'Cyber_Security_Advanced', 'Cyber_Security_Basics',
       'Data_Science_Basics', 'Data_Structures_Advanced',
       'Data_Structures_Basics', 'Databases_Advanced', 'Databases_Basics',
       'Machine_Learning_Advanced', 'Machine_Learning_Basics',
       'Mobile_Development_Advanced', 'Mobile_Development_Basics',
       'Python_Advanced', 'Python_Intro', 'Software_Engineering_Advanced',
       'Software_Engineering_Basics', 'Web_Development_Advanced',
       'Web_Development_Basics']

origins = [
    "*",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentSchema(BaseModel):
    video_view_count: int
    assignment_attempts: int
    imd_band: str
    highest_education: str
    previous_assessment_scores: List[float]

class Database:
    def __init__(self):
        pass

    def query(self, sql):
        print(f"Executing SQL query: {sql}")
        return []

db = Database()

class NBAData(BaseModel):
    label: Literal['pass','distinction']
    homepage: int
    resource: int
    quiz: int
    oucontent: int
    subpage: int
    forumng: int
    url: int
    ouwiki: int
    oucollaborate: int
    page: int
    glossary: int
    externalquiz: int

class NBAModel:
    def __init__(self):
        self.model = joblib.load('models/nba_rf_model.joblib')
        pass

    def next_best_action(self, data: NBAData):
        features = pd.DataFrame([{col : getattr(data,col)  for col in self.model.feature_names_in_}])
        prob = self.model.predict_proba(features)[0]
        goal = 1 if getattr(data, "label") == "pass" else 2
        base_prob = prob[goal]

        improvements = []
        for feature in self.model.feature_names_in_:
            modified = data.copy()
            setattr(modified,feature, getattr(modified, feature) + goal * 30)
            features = pd.DataFrame([{col : getattr(modified,col)  for col in self.model.feature_names_in_}])
            new_prob = self.model.predict_proba(features)[0][goal]
            delta = new_prob - base_prob
            improvements.append((feature, round(new_prob, 4), round(delta, 4)))

        # Sort by most positive impact
        sorted_features = sorted(improvements, key=lambda x: -x[2])
        recommendations = [{"name": feat, "change": delta} for feat, prob, delta in sorted_features if delta > 0]

        return {
            "original_score_prob": round(base_prob, 4),
            "top_resource_recommendations": recommendations or []
        }

nba_model = NBAModel()

class CourseRecommender:
    def __init__(self):
        self.model = AlternatingLeastSquares().load('models/model.npz')
        self.user_ratings = joblib.load('models/user_ratings.joblib')
        pass

    def recommend(self, userid:int):
        if (userid >= self.user_ratings.shape[0]):
            return course_names[0:3]
        ids, scores = self.model.recommend(userid, self.user_ratings[userid], N=5, filter_already_liked_items=False)
        recommended = []
        for i in ids: 
            recommended.append({
                "id": i.item(),
                "name": course_names[i],
                "ongoing": i.item() in self.user_ratings[userid].indices
            })
        return recommended


courseRecommender = CourseRecommender()

class MLModel:
    def __init__(self):
        self.tensorflow_model = None

    def predict(self, final_result_probability):
        # Placeholder prediction
        print("Predicting student performance...")
        return {"prediction": "Pass" if final_result_probability > 0.5 else "Fail"}

ml_model = MLModel()

# --- Core Feature Modules (Placeholders) ---
class ResourceEngagementHeatmap:
    def generate(self, data):
        print("Generating resource engagement heatmap with data:", data)
        return {"heatmap": "dummy_heatmap_data"}

class AssessmentScoreTrends:
    def analyze(self, data):
        print("Analyzing assessment score trends with data:", data)
        return {"trends": "dummy_trends_data"}

class PredictiveAnalyticsEngine:
    def calculate_threshold(self, q1, q3):
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        print(f"Calculated study hours threshold: {threshold}")
        return threshold

heatmap_generator = ResourceEngagementHeatmap()
trends_analyzer = AssessmentScoreTrends()
analytics_engine = PredictiveAnalyticsEngine()

# --- API Endpoints ---
@app.post('/next')
def predict_next_action(data: NBAData):
    """
    Predicts student performance based on provided data.
    """
    try:
        return nba_model.next_best_action(data)
        # return jsonify(prediction)
    except Exception as e:
        return e.__str__()

@app.get("/recommend")
def recommend_course(userid: int):
    try :
        return courseRecommender.recommend(userid)
    except Exception as e:
        return "Error occured" + e.__str__()


@app.get("/")
def dummy_root():
  return {"message": "Welcome to the LearnWise LMS API"}

