import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# โหลดโมเดลที่บันทึกไว้
with open("model/pet_model.pkl", "rb") as f:
    clf, label_encoders, acc_score = pickle.load(f)
 
st.set_page_config(page_title="Pet Recommender", page_icon="🐾", layout="centered")

# -------------------------------
# Cleancing data
# -------------------------------

df_data = pd.read_csv('data\pet_dataset_final.csv')
df_data["Budget"] = pd.cut(
    df_data["Budget"],
    bins=[-1, 1000, 3000, float("inf")],    
    labels=["Less than 1000", "Between 1000 and 3000", "Greater than 3000"]
)
df_data["PreferQuietPet"] = df_data["PreferQuietPet"].replace(1, "Yes")
df_data["PreferQuietPet"] = df_data["PreferQuietPet"].replace(0, "No")
df_data["Allergies"] = df_data["Allergies"].replace(1, "Yes")
df_data["Allergies"] = df_data["Allergies"].replace(0, "No")

df_data.to_csv("data_ready/dataset1.csv", index=False, encoding="utf-8-sig")

# New Cleancing data

df_data_forms = pd.read_csv('data\pet_dataset_form.csv')
df_data_forms = df_data_forms.drop(["ประทับเวลา", "คะแนน"], axis=1)
df_data_forms.rename(columns={"สัตว์เลี้ยงที่เลี้ยงอยู่คืออะไร": "PetChoice",
                   "เพศ":"Gender",
                   "งบประมาณในการเลี้ยงสัตว์(ต่อเดือน)":"Budget",
                   "เวลาว่างเฉลี่ยต่อวัน (24 ชั่วโมง)":"FreeTime",
                   "เวลาอยู่บ้านเฉลี่ยต่อวัน (24 ชั่วโมง)":"TimeAtHome",
                   "ที่อยู่อาศัย": "Living",
                   "สัตว์เลี้ยงเงียบหรือไม่":"PreferQuietPet",
                   "เป็นภูมแพ้หรือไม่":"Allergies",
                    },inplace=True)

# แยกคอลัมน์ PetChoice ออกมา
pet_choice = df_data_forms.pop('PetChoice')

# แทรกกลับเข้าไปหลัง Allergies
df_data_forms.insert(df_data_forms.columns.get_loc('Allergies') + 1, 'PetChoice', pet_choice)
df_data_forms['Gender'] = df_data_forms['Gender'].replace({'ชาย': 'Male', 'หญิง': 'Female'})
df_data_forms['PetChoice'] = df_data_forms['PetChoice'].replace({
    'ปลา': 'Fish',
    'สุนัข': 'Dog',
    'แมว': 'Cat',
    'แฮมเตอร์': 'Hamster',
    'งู': 'Snake',
    'ไม่เลี้ยงสัตว์': 'No pet'
})

def extract_numeric(val):
    try:
        return int(''.join(filter(str.isdigit, str(val))))
    except:
        return None

df_data_forms['FreeTime'] = df_data_forms['FreeTime'].apply(extract_numeric)
df_data_forms['TimeAtHome'] = df_data_forms['TimeAtHome'].apply(extract_numeric)

df = df_data_forms[(df_data_forms['FreeTime'] <= 24) & (df_data_forms['TimeAtHome'] <= 24)]
def extract_budget(val):
    try:
        return int(''.join(filter(str.isdigit, str(val))))
    except:
        return None

# สร้างคอลัมน์ชั่วคราวเพื่อใช้กรอง
df_data_forms['__budget_numeric'] = df_data_forms['Budget'].apply(extract_budget)
# ลบแถวที่งบประมาณเกิน 100000
df_data_forms = df_data_forms[df_data_forms['__budget_numeric'] <= 100000]
# แทนค่าในคอลัมน์ Budget เดิมด้วยช่วงที่ต้องการ
def categorize_budget(num):
    if num is None:
        return 'Unknown'
    elif num < 1000:
        return 'Less than 1000'
    elif 1000 <= num <= 3000:
        return 'Between 1000 and 3000'
    else:
        return 'Greater than 3000'

df_data_forms['Budget'] = df_data_forms['__budget_numeric'].apply(categorize_budget)
# ลบคอลัมน์ชั่วคราว
df_data_forms.drop(columns=['__budget_numeric'], inplace=True)
df_data_forms.to_csv("data_ready/dataset2.csv", index=False, encoding="utf-8-sig")

# -------------------------------
# Load datasets Used data 

df1 = pd.read_csv("data_ready\dataset2.csv").head(60)  
df2 = pd.read_csv("data_ready\dataset1.csv") 
df_combined = pd.concat([df1, df2], ignore_index=True)
output_folder = "data_ready"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "pet_dataset_merged.csv")
df_combined.to_csv(output_path, index=False)


# -------------------------------
# STEP 1: Load and Train Model
# -------------------------------

@st.cache_resource

def train_pet_model():
    df = pd.read_csv("data_ready/pet_dataset_merged.csv")

    features = ['Gender', 'Budget', 'FreeTime', 'TimeAtHome', 'Living', 'PreferQuietPet', 'Allergies']
    target = 'PetChoice'

    label_encoders = {}
    df_encoded = df.copy()
    for col in features + [target]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded[features]
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    with open("model/pet_model.pkl", "wb") as f:
        pickle.dump((clf, label_encoders, acc_score), f)


    return clf, label_encoders, acc_score

clf, label_encoders, acc_score = train_pet_model()

# -------------------------------
# STEP 2: Streamlit UI
# -------------------------------

st.markdown(
    """
    <div style="
        text-align: center; 
        background: linear-gradient(90deg, #4CAF50, #2E8B57); 
        padding: 20px; 
        border-radius: 15px; 
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        white-space: nowrap;
    ">
        <h1 style="color: white; font-size: 36px; margin: 0; line-height: 1.2;">
             ระบบแนะนำสัตว์เลี้ยงที่เหมาะกับคุณ 🐱
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

with st.expander("ℹ️ วิธีการใช้งาน"):
    st.markdown("""
    1. กรอกข้อมูลพฤติกรรมและลักษณะของคุณ  
    2. กดปุ่ม **"ทำนายสัตว์เลี้ยงที่เหมาะกับคุณ"**  
    3. ระบบจะแสดงสัตว์เลี้ยงที่เหมาะกับคุณ 3 อันดับแรก พร้อมความแม่นยำของโมเดล  

    ⚠️ หมายเหตุ: ผลลัพธ์เป็นเพียงการแนะนำเบื้องต้น ควรพิจารณาปัจจัยอื่นๆ ร่วมด้วย
    """)

st.markdown("### ✍️ กรอกข้อมูลของคุณ")

# Input Section
user_input = {}
col1, col2 = st.columns(2)

for i, col in enumerate(['Gender', 'Budget', 'Living', 'PreferQuietPet', 'Allergies']):
    with (col1 if i % 2 == 0 else col2):
        user_input[col] = st.selectbox(f"{col}", label_encoders[col].classes_)

with col1:
    user_input['FreeTime'] = st.slider("🕒 เวลาว่างต่อวัน (ชั่วโมง)", min_value=0, max_value=24, value=4)

with col2:
    user_input['TimeAtHome'] = st.slider("🏠 เวลาที่อยู่บ้านต่อวัน (ชั่วโมง)", min_value=0, max_value=24, value=8)

# -------------------------------
# STEP 3: Prediction
# -------------------------------
st.markdown("---")
if st.button("🔍 ทำนายสัตว์เลี้ยงที่เหมาะกับคุณ"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if col in ['FreeTime', 'TimeAtHome']:
            continue
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_df = input_df[clf.feature_names_in_]

    probs = clf.predict_proba(input_df)[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3_pets = [label_encoders['PetChoice'].inverse_transform([i])[0] for i in top3_idx]

    # Show accuracy
    st.metric("📊 ความแม่นยำของโมเดล", f"{acc_score*100:.2f}%")
    # Show results in cards
    st.success("🐾 ผลการทำนาย: สัตว์เลี้ยงที่เหมาะกับคุณ 3 อันดับแรก")
    for i, pet in enumerate(top3_pets, 1):
        st.markdown(
    f"""
    <div style="
        background-color:#2E2E2E; 
        padding:15px; 
        margin:10px 0; 
        border-radius:12px; 
        border-left:6px solid #4CAF50;
        color:#FFFFFF;">
        <h4 style="margin:0;">{i}. {pet}</h4>
    </div>
    """, 
    unsafe_allow_html=True
)
        
with st.sidebar:
    st.header("ℹ️ วิธีการใช้งานแบบละเอียด")

    with st.expander("วิธีการใช้งานแบบละเอียด"):
        st.write("""
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

    st.header("📖 คู่มือการเลี้ยงสัตว์")

    with st.expander("🐶 การเลี้ยงสุนัข (Dog)"):
        st.write("**1.เรื่องอาหาร**\n"
        "- ให้อาหารสุนัขวันละ 1–2 มื้อ (ขึ้นอยู่กับอายุและขนาด)\n"
        "- เลือกอาหารเม็ดหรืออาหารปรุงเองที่เหมาะกับสายพันธุ์และวัย\n"
        "- ห้ามให้ของหวาน ช็อกโกแลต องุ่น หัวหอม กระเทียม เพราะเป็นพิษกับสุนัข\n" 
        "- ควรมีน้ำสะอาดให้ตลอดเวลา\n" 

        "\n**2. การดูแลสุขภาพ**\n"
        "- พาไปฉีดวัคซีนและถ่ายพยาธิตามกำหนด\n"
        "- หมั่นดูแลเรื่องหมัด เห็บ และพาไปพบสัตวแพทย์เมื่อมีอาการผิดปกติ\n" \
        
        "\n**3. การออกกำลังกายและกิจกรรม**\n"
        "- พาออกไปเดินเล่นหรือวิ่งเล่นทุกวัน เล่นกับน้องหมาเพื่อให้เขาไม่เครียดและผูกพันกับเจ้าของ\n"
        
        "\n**4. การเลี้ยงดูด้านจิตใจ**\n" 
        "- ให้ความรัก เอาใจใส่ และฝึกวินัย ไม่ใช้ความรุนแรง ควรใช้วิธี ให้รางวัล เมื่อทำถูกต้อง\n" 
        "\n**5. ที่อยู่อาศัย**\n" 
        "- จัดที่นอนสะอาด แห้ง และอากาศถ่ายเท\n" 
        "- ไม่ควรผูกโซ่นานเกินไป ควรมีพื้นที่ให้วิ่งเล่น\n"

        "")
    with st.expander("🐱 การเลี้ยงแมว (Cat)"):
        st.write("**1. เรื่องอาหาร**\n"
        "- ให้อาหารแมววันละ 2–3 มื้อ หรือเลือกให้อาหารเม็ดแบบ แบ่งปริมาณไว้ทั้งวัน\n"
        "- ห้ามให้ ช็อกโกแลต หัวหอม กระเทียม องุ่น นมวัว เพราะเป็นพิษต่อแมว\n"
        "- แมวบางตัวชอบน้ำไหล ควรมี น้ำพุแมว เพื่อกระตุ้นให้ดื่มมากขึ้น\n" 

        "\n**2. การดูแลสุขภาพ**\n"
        "- ฉีดวัคซีนและถ่ายพยาธิตามกำหนด\n"
        "- ทำความสะอาด กระบะทราย ทุกวัน และเปลี่ยนทรายอย่างน้อยสัปดาห์ละครั้ง\n" 
        
        "\n**3. การเลี้ยงดูด้านจิตใจ**\n" 
        "- เล่นกับแมววันละ 15–30 นาที ด้วยของเล่น เช่น ไม้ล่อแมว ลูกบอล"
        "- ให้ความรักและเอาใจใส่ แต่ก็เคารพความเป็นส่วนตัวของแมว (เพราะแมวชอบมีโลกส่วนตัว)" 
        
        "\n**4. ที่อยู่อาศัย**\n" 
        "- แมวชอบพื้นที่สงบ ควรมีที่นอนและที่ซ่อน\n" 
        "- จัดมุมปีนป่าย เช่น คอนโดแมว หรือชั้นวางของสูง ๆ\n"
        "- บ้านควรปลอดภัย ไม่ควรให้วิ่งเล่นนอกบ้านเพราะเสี่ยงอุบัติเหตุและโรค\n"

        "")
    with st.expander("🐠 การเลี้ยงปลา (Fish)"):
        st.write("**1. การเตรียมตู้ปลา**\n"
        "- เลือกตู้ให้เหมาะกับชนิดและจำนวนปลา\n"
        "- ใส่ กรวด หิน พืชน้ำ หรือของตกแต่ง ให้ปลาได้หลบซ่อน\n"
        "- ต้องมี เครื่องกรองน้ำ เพื่อรักษาความสะอาด และ ปั๊มอากาศ เพื่อเพิ่มออกซิเจน\n" 

        "\n**2. น้ำและสิ่งแวดล้อม**\n"
        "- ใช้น้ำที่ปราศจากคลอรีน (พักน้ำอย่างน้อย 24 ชม. หรือใช้น้ำยาลดคลอรีน)\n"
        "- เปลี่ยนน้ำบางส่วน 20–30% ทุกสัปดาห์ ไม่ควรเปลี่ยนทั้งหมดทีเดียว\n" 
        
        "\n**3. อาหาร**\n" 
        "- ให้อาหารเม็ดหรืออาหารสดที่เหมาะกับสายพันธุ์ปลา ให้ในปริมาณที่กินหมดภายใน 2–3 นาที วันละ 1–2 ครั้ง"
        "- ไม่ควรให้อาหารมากเกินไปเพราะจะทำให้น้ำเสีย" 
        
        "\n**4. การดูแลสุขภาพปลา**\n" 
        "- สังเกตอาการผิดปกติ เช่น ว่ายน้ำไม่ปกติ ตัวซีด มีจุดขาว (อาจติดเชื้อ)\n" 
        "- แยกปลาป่วยออกมาต่างหากเพื่อป้องกันการแพร่โรค\n"
        "- รักษาคุณภาพน้ำให้ดีเสมอ เพราะสุขภาพปลาขึ้นอยู่กับน้ำ\n"

        "\n**5. การเลี้ยงที่เหมาะสม**\n" 
        "- ไม่ควรเลี้ยงปลาหลายชนิดที่ดุร้ายรวมกัน เพราะอาจกัดกันได้\n" 
       

        "")
    with st.expander("🐹 การเลี้ยงแฮมสเตอร์ (Hamster)"):
        st.write("**1. กรงและที่อยู่อาศัย**\n"
        "- เลือกกรงที่มี พื้นโปร่ง ระบายอากาศได้ดี ไม่ควรเป็นตู้ปิดทึบ\n"
        "- ปูรองพื้นด้วย ขี้เลื่อยไม้เนื้ออ่อน (ห้ามใช้ไม้สน/ซีดาร์ เพราะมีสารระเหยอันตราย)\n"
        "- ควรมี ล้อวิ่ง สำหรับออกกำลังกาย\n" 

        "\n**2. อาหาร**\n"
        "- อาหารหลักคือ อาหารเม็ดสำหรับแฮมสเตอร์ และสามารถเสริมด้วย ผักสด ผลไม้ (เช่น แครอท แอปเปิ้ล) แต่ต้องให้ในปริมาณน้อย\n"
        "- ห้ามให้ ช็อกโกแลต ของหวาน ของเค็ม เพราะเป็นอันตราย \n" 
        
        "\n**3. สุขภาพและการดูแล**\n" 
        "- ทำความสะอาดกรงสัปดาห์ละ 1–2 ครั้ง\n"
        "- ไม่ควรอาบน้ำด้วยน้ำจริง แต่ใช้ ทรายอาบแห้งสำหรับแฮมสเตอร์ ให้เขากลิ้งเล่นเพื่อทำความสะอาดตัวเอง\n" 
        "- สังเกตพฤติกรรม หากซึม ไม่กินอาหาร หรือมีอาการผิดปกติ ควรพาไปพบสัตวแพทย์\n" 

        
        "\n**4. พฤติกรรมและการเล่น**\n" 
        "- แฮมสเตอร์ส่วนใหญ่เป็นสัตว์หวงถิ่น ควรเลี้ยงเดี่ยว 1 กรงต่อ 1 ตัว\n" 
        "- เล่นกับเขาเบา ๆ อย่าจับแรง เพราะตัวเล็กและกระดูกเปราะ\n"
        "- ควรฝึกให้คุ้นมือเจ้าของด้วยการค่อย ๆ ยื่นอาหารให้\n"

        "")
    with st.expander("🐍 การเลี้ยงงู (Snake)"):
        st.write("**1. ที่อยู่อาศัย**\n"
        "- ใช้ตู้กระจกหรือกล่องเลี้ยงงูที่ปิดมิดชิด ป้องกันงูเลื้อยออกมาได้\n"
        "- ควรมีพื้นที่ให้หลบซ่อน เช่น กล่องหรือหินจำลอง\n"
        "- รักษาอุณหภูมิที่เหมาะสม (ส่วนใหญ่ 25–30°C) และมีโคมไฟ/แผ่นความร้อนช่วย\n"
        "- ปูพื้นด้วยวัสดุ เช่น ขี้เลื่อย กระดาษ หรือทราย (แล้วแต่ชนิดงู)\n"

        "\n**2. อาหาร**\n"
        "- อาหารหลักคือ หนูแช่แข็ง/สัตว์ฟันแทะที่เหมาะสมกับขนาดงู\n"
        "- ให้อาหารทุก 1–2 สัปดาห์ ขึ้นอยู่กับอายุและขนาด\n"
        "- ไม่ควรจับหรือลูบคลำหลังให้อาหารทันที เพราะงูอาจตื่นตกใจและสำรอกได้\n"

        "\n**3. สุขภาพและการดูแล**\n"
        "- คอยสังเกตอาการผิดปกติ เช่น ไม่กินอาหาร ผิวซีด มีรอยโรค หรือหายใจเสียงดัง\n"
        "- ทำความสะอาดตู้และเปลี่ยนน้ำอย่างน้อยสัปดาห์ละครั้ง\n"
        "- ช่วงลอกคราบควรรักษาความชื้นให้เพียงพอ\n"

        "\n**4. ความปลอดภัย**\n"
        "- ควรศึกษาสายพันธุ์ก่อนเลี้ยง บางชนิดดุร้าย ไม่เหมาะกับมือใหม่\n"
        "- ห้ามเลี้ยงงูพิษถ้าไม่มีประสบการณ์และอุปกรณ์พร้อม\n"
        "- เวลาจับควรใช้ที่คีบหรือตะขอช่วย และจับด้วยความระมัดระวัง\n"

        "")
    with st.expander("ไม่ควรเลี้ยงสัตว์ (No Pet)"):
            st.write("**ไม่ควรเลี้ยงสัตว์เนื่องจากคุณสมบัติไม่พร้อม**\n")