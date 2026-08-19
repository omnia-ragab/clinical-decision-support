# 🏥 Clinical Decision Support - Streamlit App

**Evidence-based Clinical Recommendations using RAG + Gemini API**

---

## 🎯 اختار طريقة التشغيل

### **الخيار 1: Hugging Face Spaces (الأسهل) ⭐⭐⭐**

**الخطوات:**

1. اذهب إلى https://huggingface.co/spaces
2. اضغط **"Create new Space"**
3. اختر:
   - **Owner:** اختر حسابك
   - **Space name:** `clinical-decision-support`
   - **License:** اختر أي واحدة
   - **Space SDK:** اختر **Streamlit**
   - **Visibility:** Public

4. في الصفحة الجديدة، اضغط **"Create Space"**

5. **احط الملفات:**
   - انسخ محتوى `app.py` → الصق في ملف جديد `app.py`
   - انسخ محتوى `requirements.txt` → الصق في ملف جديد `requirements.txt`

6. اضغط **"Save and Run"**

7. **تمام! ✅** الـ app هيرفع تلقائياً في رابط مثل:
   ```
   https://huggingface.co/spaces/[username]/clinical-decision-support
   ```

---

### **الخيار 2: Streamlit Cloud (سهل جداً)**

**الخطوات:**

1. **أنشئ GitHub repo:**
   - اذهب إلى github.com
   - اضغط **"New repository"**
   - اسم الـ repo: `clinical-decision-support`
   - اضغط **"Create repository"**

2. **رفع الملفات:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/clinical-decision-support.git
   cd clinical-decision-support
   
   # انسخ app.py و requirements.txt هنا
   
   git add .
   git commit -m "Add Streamlit app"
   git push
   ```

3. **Deploy على Streamlit Cloud:**
   - اذهب إلى https://share.streamlit.io
   - اضغط **"New app"**
   - اختر GitHub repo و branch و file (`app.py`)
   - اضغط **"Deploy"**

4. **تمام! ✅** الـ app هيرفع في رابط مثل:
   ```
   https://[username]-clinical-decision-support.streamlit.app
   ```

---

### **الخيار 3: تشغيل Local (جهازك)**

**الخطوات:**

```bash
# 1. تحميل الملفات
git clone <repo-url>
cd clinical-decision-support

# 2. إنشاء virtual environment (اختياري)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# أو
venv\Scripts\activate     # Windows

# 3. تثبيت المكتبات
pip install -r requirements.txt

# 4. تشغيل الـ app
streamlit run app.py

# 5. فتح البراوزر في:
# http://localhost:8501
```

---

## 🔑 إضافة API Keys

### **للـ Gemini API:**

1. اذهب إلى https://makersuite.google.com/app/apikey
2. اضغط **"Create API Key"**
3. انسخ الـ key

**في Hugging Face Spaces:**
- اذهب إلى Settings → Secrets
- أضف secret جديد:
  ```
  GEMINI_API_KEY = "your-key-here"
  ```

**في Streamlit Cloud:**
- اذهب إلى App settings → Secrets
- أضف:
  ```
  GEMINI_API_KEY = "your-key-here"
  ```

**Locally:**
- أنشئ ملف `.streamlit/secrets.toml`:
  ```
  GEMINI_API_KEY = "your-key-here"
  ```

---

## 📊 الميزات الأساسية

- ✅ **Ask Question** - اسأل أسئلة طبية
- ✅ **Query History** - شوف الأسئلة السابقة
- ✅ **Sources** - شوف الـ guidelines المستخدمة
- ✅ **About** - معلومات عن النظام
- ✅ **Settings** - إعدادات متقدمة

---

## 🏗️ بنية المشروع

```
clinical-decision-support/
├── app.py                    # الـ app الرئيسي
├── requirements.txt          # المكتبات المطلوبة
├── README.md                 # هذا الملف
└── .streamlit/
    └── secrets.toml          # API keys (في local فقط)
```

---

## 🚀 الطريقة الأسرع (5 دقائق فقط)

### **استخدم Hugging Face Spaces:**

1. اذهب إلى https://huggingface.co/spaces
2. **Create new Space** → Streamlit
3. **Upload:** app.py + requirements.txt
4. **Save** → الـ app يرفع تلقائياً ✅
5. **Share الرابط!**

---

## ⚠️ تنبيهات مهمة

- هذا النظام **يدعم** اتخاذ القرارات الطبية، **لا يستبدل** الحكم السريري
- استخدم دائماً الـ guidelines الرسمية
- تحقق من التوصيات مع الخبراء
- الـ app يعمل أفضل مع Gemini API (مجاني للـ testing)

---

## 📞 دعم

إذا حصلت مشكلة:

1. **قراءة الأخطاء في Logs** - أهم حاجة
2. **تأكد من API Keys** - غالب المشاكل من هنا
3. **تحديث المكتبات:**
   ```bash
   pip install --upgrade streamlit google-generativeai
   ```

---

## 📝 ملاحظات تطوير

- **للـ integration مع vector DB:** عدّل `app.py` وأضف الـ retrieval logic
- **للـ Gemini API:** استخدم `google.generativeai` library
- **للـ ChromaDB:** أضف local embeddings في الـ app
- **للـ PDF parsing:** استخدم LlamaParse في pipeline

---

**Made with ❤️ for AI Hackathons**

🏥 Clinical Decision Support System | Evidence-based • Traceable • Trustworthy
