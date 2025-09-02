# 🐾 Pet Recommendation System

ระบบแนะนำสัตว์เลี้ยงที่เหมาะกับผู้ใช้งาน โดยใช้ Machine Learning แบบ Supervised Learning (Decision Tree) และพัฒนาอินเทอร์เฟซด้วย Streamlit

## 🔍 รายละเอียดโปรเจกต์

ผู้ใช้สามารถกรอกข้อมูลเกี่ยวกับพฤติกรรมและไลฟ์สไตล์ เช่น:
- เพศ
- งบประมาณรายเดือน
- เวลาว่าง
- เวลาอยู่บ้าน
- ประเภทที่อยู่อาศัย
- ความต้องการสัตว์เลี้ยงที่เงียบ
- อาการแพ้สัตว์เลี้ยง

ระบบจะทำการทำนายสัตว์เลี้ยงที่เหมาะสมที่สุด 3 อันดับแรก พร้อมแสดงความแม่นยำของโมเดล

## 📁 โครงสร้างไฟล์
ใช้ไฟล์ชื่อ pet_dataset_merged.csv

## 🚀 วิธีใช้งาน

1. ติดตั้งไลบรารีที่จำเป็น:
   ```bash
   pip install -r requirements.txt
   
2. รันแอป Streamlit:
    streamlit run test.py

3. กรอกข้อมูลผ่านหน้าเว็บ และดูผลการแนะนำสัตว์เลี้ยง
📊 โมเดลที่ใช้- DecisionTreeClassifier จาก scikit-learn
- LabelEncoder สำหรับจัดการข้อมูลหมวดหมู่
- Accuracy ถูกคำนวณจากชุดทดสอบ (test set)
🧠 ข้อมูลที่ใช้ไฟล์ pet_dataset_merged.csv ต้องมีคอลัมน์ดังนี้:- Gender
- Budget
- FreeTime
- TimeAtHome
- Living
- PreferQuietPet
- Allergies
- PetChoice

## 🙋‍♂️ ผู้พัฒนา
นายสุทธิชัย มุกโชควัฒนา
นายนพวิชญ์ ไชยรต