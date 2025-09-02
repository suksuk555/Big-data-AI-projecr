import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





# -------------------------------
# STEP 1: Load and Train Model
# -------------------------------
@st.cache_resource
def train_pet_model():
    df = pd.read_csv("data_ready/pet_dataset_merged.csv")

    # เลือกเฉพาะคอลัมน์ที่ใช้
    features = ['Gender', 'Budget', 'FreeTime', 'TimeAtHome', 'Living', 'PreferQuietPet', 'Allergies']
    target = 'PetChoice'

    # สร้าง LabelEncoder สำหรับแต่ละคอลัมน์
    label_encoders = {}
    df_encoded = df.copy()
    for col in features + [target]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded[features]
    y = df_encoded[target]

    # เทรนโมเดล Decision Tree
    # clf = DecisionTreeClassifier(random_state=42)
    # clf.fit(X, y)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    return clf, label_encoders, acc_score


# โหลดโมเดลและ encoder
clf, label_encoders, acc_score = train_pet_model()

# -------------------------------
# STEP 2: Streamlit Interface
# -------------------------------
st.title("🐶 ระบบแนะนำสัตว์เลี้ยงที่เหมาะกับคุณ")
if "show_help" not in st.session_state:
    st.session_state.show_help = False
if st.button("ℹ️ ข้อมูลการใช้งาน"):
    st.session_state.show_help = not st.session_state.show_help
if st.session_state.show_help:
    st.info("""
    1. กรอกข้อมูลพฤติกรรมและลักษณะของคุณ \n
        -Gender(เพศ) : เพศของผู้เลี้ยงสัตว์เลี้ยง \n
        -Budget(งบประมาณ) : งบประมาณที่มีต่อการเลี้ยงสัตว์เลี้ยงต่อเดือน  \n
        -Living(ที่อยู่อาศัย) : ประเภทที่อยู่อาศัย เช่น อพาร์ตเมนต์ คอนโด บ้าน \n
        -PreferQuietPet(ชอบสัตว์เลี้ยงที่เงียบไหม) : ความชอบส่วนตัวเกี่ยวกับเสียงของสัตว์เลี้ยง \n
        -Allergies(แพ้ขนสัตว์ไหม) : ข้อมูลเกี่ยวกับการแพ้ขนสัตว์ ซึ่งมีผลต่อการเลือกประเภทสัตว์เลี้ยง \n
        -FreeTime(เวลาว่างต่อวัน) : เวลาว่างที่มีต่อการดูแลสัตว์เลี้ยงต่อวัน \n
        -TimeAtHome(เวลาที่อยู่บ้านต่อวัน) : เวลาที่อยู่บ้านต่อวัน ซึ่งมีผลต่อการดูแลสัตว์เลี้ยง \n
    2. กดปุ่ม "ทำนายสัตว์เลี้ยงที่เหมาะกับคุณ" \n
    3. ระบบจะแสดงสัตว์เลี้ยงที่เหมาะกับคุณ 3 อันดับแรก พร้อมความแม่นยำของโมเดล   \n
    หมายเหตุ: ผลลัพธ์ที่ได้เป็นเพียงการแนะนำเบื้องต้น ควรพิจารณาปัจจัยอื่นๆ ร่วมด้วยก่อนตัดสินใจเลี้ยงสัตว์
            
    
    """)

st.markdown("กรอกข้อมูลด้านล่างเพื่อดูว่าสัตว์เลี้ยงแบบไหนเหมาะกับคุณที่สุด")

# รับข้อมูลจากผู้ใช้
user_input = {}
col1, col2 = st.columns(2)

for i, col in enumerate(['Gender', 'Budget', 'Living', 'PreferQuietPet', 'Allergies']):
    with (col1 if i % 2 == 0 else col2):
        user_input[col] = st.selectbox(f"{col}", label_encoders[col].classes_)

# เพิ่ม slider สำหรับ FreeTime และ TimeAtHome
with col1:
    user_input['FreeTime'] = st.slider("FreeTime (ชั่วโมง)", min_value=0, max_value=24, value=4)

with col2:
    user_input['TimeAtHome'] = st.slider("TimeAtHome (ชั่วโมง)", min_value=0, max_value=24, value=8)
# -------------------------------
# STEP 3: Predict and Show Result
# -------------------------------
if st.button("🔍 ทำนายสัตว์เลี้ยงที่เหมาะกับคุณ"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if col in ['FreeTime', 'TimeAtHome']:
            continue  # ไม่ต้อง encode ค่าตัวเลข
        input_df[col] = label_encoders[col].transform(input_df[col])

    # จัดลำดับคอลัมน์ให้ตรงกับตอนเทรน
    input_df = input_df[clf.feature_names_in_]

    # ทำนายความน่าจะเป็น
    probs = clf.predict_proba(input_df)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_pets = [label_encoders['PetChoice'].inverse_transform([i])[0] for i in top3_idx]

    st.metric("📊 ความแม่นยำของโมเดล", f"{acc_score*100:.2f}%")

    st.success("🐾 สัตว์เลี้ยงที่เหมาะกับคุณ 3 อันดับแรกคือ:")
    for i, pet in enumerate(top3_pets, 1):
        st.write(f"{i}. {pet}")

