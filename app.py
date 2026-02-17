import streamlit as st
import pandas as pd
import joblib
import altair as alt

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

model = load_model()
feature_columns = model.named_steps["preprocess"].feature_names_in_

# ---------------------------------
# Thai Labels
# ---------------------------------
thai_labels = {
    "Marital status": "สถานภาพสมรส",
    "Application mode": "รูปแบบการสมัครเข้าเรียน",
    "Application order": "ลำดับการสมัคร",
    "Course": "สาขาวิชาที่สมัคร",
    "Daytime/evening attendance": "รูปแบบการเรียน (ภาคปกติ/ภาคค่ำ)",
    "Previous qualification": "วุฒิการศึกษาก่อนหน้า",
    "Previous qualification (grade)": "เกรดเฉลี่ยวุฒิการศึกษาก่อนหน้า",
    "Nacionality": "สัญชาติ",
    "Mother's qualification": "ระดับการศึกษาของมารดา",
    "Father's qualification": "ระดับการศึกษาของบิดา",
    "Mother's occupation": "อาชีพของมารดา",
    "Father's occupation": "อาชีพของบิดา",
    "Admission grade": "คะแนนเฉลี่ยตอนเข้าศึกษา",
    "Displaced": "นักศึกษาย้ายถิ่นฐาน",
    "Educational special needs": "มีความต้องการทางการศึกษาพิเศษ",
    "Debtor": "มีสถานะค้างชำระค่าเล่าเรียน",
    "Tuition fees up to date": "ชำระค่าเล่าเรียนครบถ้วน",
    "Scholarship holder": "ได้รับทุนการศึกษา",
    "Gender": "เพศ",
    "Age at enrollment": "อายุเมื่อเข้าเรียน",
    "International": "นักศึกษาต่างชาติ",
    "Unemployment rate": "อัตราการว่างงาน",
    "Inflation rate": "อัตราเงินเฟ้อ",
    "GDP": "ผลิตภัณฑ์มวลรวมภายในประเทศ",
    "Curricular units 1st sem (credited)": "หน่วยกิตเทียบโอน ภาคเรียนที่ 1",
    "Curricular units 1st sem (enrolled)": "หน่วยกิตลงทะเบียน ภาคเรียนที่ 1",
    "Curricular units 1st sem (evaluations)": "จำนวนครั้งประเมินผล ภาคเรียนที่ 1",
    "Curricular units 1st sem (approved)": "หน่วยกิตผ่าน ภาคเรียนที่ 1",
    "Curricular units 1st sem (grade)": "เกรดเฉลี่ย ภาคเรียนที่ 1",
    "Curricular units 1st sem (without evaluations)": "วิชาไม่มีประเมิน ภาคเรียนที่ 1",
    "Curricular units 2nd sem (credited)": "หน่วยกิตเทียบโอน ภาคเรียนที่ 2",
    "Curricular units 2nd sem (enrolled)": "หน่วยกิตลงทะเบียน ภาคเรียนที่ 2",
    "Curricular units 2nd sem (evaluations)": "จำนวนครั้งประเมินผล ภาคเรียนที่ 2",
    "Curricular units 2nd sem (approved)": "หน่วยกิตผ่าน ภาคเรียนที่ 2",
    "Curricular units 2nd sem (grade)": "เกรดเฉลี่ย ภาคเรียนที่ 2",
    "Curricular units 2nd sem (without evaluations)": "วิชาไม่มีประเมิน ภาคเรียนที่ 2",
}

boolean_fields = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "International"
]

# ---------------------------------
# Header
# ---------------------------------
st.title("🎓 Student Dropout Prediction Dashboard")
st.markdown("### ระบบประเมินความเสี่ยงการลาออกของนักศึกษา")

st.divider()

# ---------------------------------
# Input Section
# ---------------------------------
input_dict = {}

sections = {
    "👤 ข้อมูลทั่วไป": [],
    "🎓 ข้อมูลการศึกษา": [],
    "💰 สถานะทางการเงิน": [],
    "🌍 ปัจจัยเศรษฐกิจ": []
}

for col in feature_columns:
    if col in ["Unemployment rate", "Inflation rate", "GDP"]:
        sections["🌍 ปัจจัยเศรษฐกิจ"].append(col)
    elif col in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
        sections["💰 สถานะทางการเงิน"].append(col)
    elif "Curricular" in col or "Admission" in col or "Previous" in col:
        sections["🎓 ข้อมูลการศึกษา"].append(col)
    else:
        sections["👤 ข้อมูลทั่วไป"].append(col)

for section, cols in sections.items():

    if not cols:
        continue

    st.subheader(section)
    col1, col2 = st.columns(2)

    for i, col in enumerate(cols):

        label = f"**{col}**"
        if col in thai_labels:
            label += f"  \n_{thai_labels[col]}_"

        if col in boolean_fields:
            value = st.selectbox(label, [0, 1])
        else:
            value = st.number_input(label, value=0.0)

        input_dict[col] = value

    st.divider()

# ---------------------------------
# Prediction Section (3-Class)
# ---------------------------------
if st.button("🔍 วิเคราะห์ผล", use_container_width=True):

    input_data = pd.DataFrame([input_dict])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.named_steps["model"].classes_

    prob_dict = dict(zip(classes, probabilities))

    dropout_prob = prob_dict.get("Dropout", 0) * 100
    enrolled_prob = prob_dict.get("Enrolled", 0) * 100
    graduate_prob = prob_dict.get("Graduate", 0) * 100

    st.subheader("📊 ผลการประเมิน")

    c1, c2, c3 = st.columns(3)

    c1.metric("🔴 Dropout", f"{dropout_prob:.2f}%")
    c2.metric("🟡 Enrolled", f"{enrolled_prob:.2f}%")
    c3.metric("🟢 Graduate", f"{graduate_prob:.2f}%")

    st.divider()

    # Risk Decision
    if dropout_prob < 20:
        st.success("🟢 ความเสี่ยงต่ำ (Low Risk)")
    elif dropout_prob < 50:
        st.warning("🟡 ความเสี่ยงปานกลาง (Medium Risk)")
    else:
        st.error("🔴 ความเสี่ยงสูง (High Risk)")

    # Bar Chart
    chart_data = pd.DataFrame({
        "Status": ["Dropout", "Enrolled", "Graduate"],
        "Probability": [dropout_prob, enrolled_prob, graduate_prob]
    })

    chart = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y("Status:N", sort="-x"),
        x=alt.X("Probability:Q", title="เปอร์เซ็นต์ (%)"),
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=["Dropout", "Enrolled", "Graduate"],
                range=["#ff4b4b", "#f0ad4e", "#28a745"]
            ),
            legend=None
        )
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.write("Predicted Class:", prediction)
    st.write("Raw probability vector:", probabilities)
