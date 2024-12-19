import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from classifier import FoodClassifier

class PredictionRequest(BaseModel):
    userId: str
    image: str
    allergy: list

app = FastAPI()

url_recommend_pop = "https://recommend-api-production.up.railway.app/recommend/popularity"  # GET method
url_recommend_cb = "https://recommend-api-production.up.railway.app/recommend/cb"  # GET method
url_recommend_cf = "https://recommend-api-production.up.railway.app/recommend/cf"  # POST method
url_allergy = "https://recommend-api-production.up.railway.app/allergy/food"  # POST method

model_path = 'model.pt' 
class_file = 'classes.txt'
foods = ["น้ำพริกปลาทู", "ขนมครก", "ผัดไทย", "สะเต๊ะไก่", "ทอดมันปลา", "น้ำตกหมู", "ผัดซีอิ๊ว", "ขนมต้ม", "เต้าฮวยน้ำขิง", "แกงมัสมั่น"]
english = ["nam_prik_pla_too", "khanom_krok", "pad_thai", "satay_gai", "tod_mun_pla", "nam_tok_moo", "pad_see_ew", "khanom_tom", "tao_huay", "massaman"]
classifier = FoodClassifier(model_path=model_path, class_file=class_file, foods=foods)

@app.post("/predict/")
async def predict(request: PredictionRequest):
    food_name = classifier.classify(request.image)
    index = foods.index(food_name)
    classified_food = english[index]
    
    async with httpx.AsyncClient() as client:
        pop_response = await client.get(url_recommend_pop)
        # cb_response = await client.get(url_recommend_cb)
        # cf_response = await client.post(url_recommend_cf, json={"food_name": classified_food})
        allergy_response = await client.post(url_allergy, json={"food_name": classified_food})

        recommendations_pop = pop_response.json()
        # recommendations_cb = cb_response.json()
        # recommendations_cf = cf_response.json()
        allergy_info = allergy_response.json()

    waring = False
    allergies = request.allergy
    for a in allergies:
        if a in allergy_info["allergy"]:
            waring = True
            break

    return {
        "userId": request.userId,
        "food": classified_food,
        "allergy": request.allergy,
        "warning": waring,
        "allergy_info": allergy_info["allergy"],
        "suggestion": recommendations_pop["best_restaurant"]
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Classifier API"}
