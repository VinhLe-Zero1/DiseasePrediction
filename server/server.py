from flask import Flask, request, jsonify
from joblib import load
import json
import requests
import numpy as np
import ast

from config import MODEL_PATH

app = Flask(__name__)

diseases = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
       'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
       'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
       'Common Cold', 'Dengue', 'Diabetes ',
       'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
       'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
       'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
       'Osteoarthristis', 'Paralysis (brain hemorrhage)',
       'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
       'Typhoid', 'Urinary tract infection', 'Varicose veins',
       'hepatitis A']

diseasesVN = ['Chóng mặt Lành tính do Tư thế ', 'AIDS', 'Nổi mụn',
              'Gan nhiễm mỡ không do rượu', 'Dị ứng', 'Viêm khớp', 'Hen phế quản',
              'Thoái hóa cột sống cổ', 'Thủy đậu', 'Ứ mật mãn tính',
              'Cảm thông thường', 'Sốt xuất huyết', 'Béo phì',
              'Trĩ hỗn hợp', 'Dị ứng thuốc', 'Nhiễm trùng nấm',
              'Bệnh trào ngược dạ dày thực quản', 'Viêm dạ dày ruột hoặc tiêu chảy nhiễm trùng', 'Đột quỵ',
              'Viêm gan B', 'Viêm gan C', 'Viêm gan D', 'Viêm gan E',
              'Cao huyết áp', 'Cường giáp', 'Hạ đường huyết',
              'Suy giáp', 'Bệnh chốc lở', 'Vàng da', 'Sốt rét', 'Đau nửa đầu',
              'Viêm khớp', 'Tê liệt (xuất huyết não)',
              'Loét dạ dày tá tràng', 'Phế cầu khuẩn', 'Vẩy nến', 'Lao',
              'Thương hàn', 'Nhiễm trùng đường tiểu', 'Giãn tĩnh mạch',
              'Viêm gan A'
              ]

with open("../data/symptom.json", "r") as f:
       jsonStr = f.read()
symptomDict = json.loads(jsonStr)

clf = load(MODEL_PATH)


@app.route('/', methods=['POST'])
def hello():
       symptom = str.split(request.get_json(force=True)['symptom'], sep=",")
       disease = np.zeros(len(symptomDict.keys()))
       for sym in symptom:
              disease[symptomDict[sym]] = 1
       result = clf.predict([disease])
       return {"result": diseases[result]}


if __name__ == '__main__':
    app.run(debug=True)