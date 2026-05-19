import os
import io
import json
import random
import requests
import gdown
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO
# =========================================
# FASTAPI
# =========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# YOUTUBE API
# =========================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# =========================================
# DOWNLOAD DOG MODEL
# =========================================

DOG_MODEL_PATH = "dog_model.keras"

if not os.path.exists(DOG_MODEL_PATH):

    gdown.download(
        url="https://drive.google.com/file/d/15s4lneWlkWg_Acf2NE5szuIZkBXnR3bl/view?usp=drive_link",
        output=DOG_MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

# =========================================
# DOWNLOAD CAT MODEL
# =========================================

CAT_MODEL_PATH = "cat_model.keras"

if not os.path.exists(CAT_MODEL_PATH):

    gdown.download(
        url="https://drive.google.com/file/d/1IauPJI2NbPSwlQ2giJO3z3nqtHI33ifh/view?usp=sharing",
        output=CAT_MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

# =========================================
# LOAD MODELS
# =========================================

dog_model = None
cat_model = None
yolo_model = None

def get_dog_model():
    global dog_model
    if dog_model is None:
        dog_model = tf.keras.models.load_model(DOG_MODEL_PATH)
    return dog_model

def get_cat_model():
    global cat_model
    if cat_model is None:
        cat_model = tf.keras.models.load_model(CAT_MODEL_PATH)
    return cat_model

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")
    return yolo_model

# =========================
# YOLO DETECTION
# =========================

def detect_animal_yolo(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    model = get_yolo_model()
    results = model(img, verbose=False)

    detected = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls_id]

            if name in ["dog", "cat"]:
                detected.append({
                    "animal": name,
                    "confidence": conf
                })

    if not detected:
        return None

    detected = sorted(
        detected,
        key=lambda x: x["confidence"],
        reverse=True
    )

    return detected[0]


# =========================================
# CLASS LABELS
# =========================================

dog_class_labels = [
    "angry",
    "happy",
    "relaxed",
    "sad"
]

cat_class_labels = [
    "angry",
    "relaxed",
    "sad"
]

# =========================================
# CAT RULES
# =========================================

def apply_cat_behavior_rules(prediction, class_labels):

    adjusted = prediction.copy()

    probs = {
        class_labels[i]: float(prediction[i])
        for i in range(len(class_labels))
    }

    angry = probs.get("angry", 0)

    if angry < 0.70:

        for i, cls in enumerate(class_labels):

            if cls == "angry":
                adjusted[i] *= 0.50

            elif cls == "relaxed":
                adjusted[i] *= 0.90

            elif cls == "sad":
                adjusted[i] *= 1.40

    adjusted = adjusted / adjusted.sum()

    pred_index = int(np.argmax(adjusted))

    emotion = class_labels[pred_index]
    confidence = float(adjusted[pred_index])

    return emotion, confidence, adjusted

# =====================================
# TIPS
# =====================================

cat_tips = {

    "angry": [
        "إذا بدت القطة غاضبة أو متوترة، لا تحاول حملها مباشرة أو إجبارها على التفاعل. امنحها مساحة آمنة وهادئة حتى تهدأ تدريجيًا، لأن الاقتراب الزائد أثناء التوتر قد يزيد العدائية والخوف.",
        
        "غالبًا ما تصبح القطط عدوانية عندما تشعر بعدم الأمان أو عند وجود أصوات مزعجة أو تغييرات مفاجئة في البيئة. حاول تقليل الضوضاء وتوفير مكان مريح يحتوي على سريرها وألعابها المفضلة.",
        
        "راقب لغة الجسد جيدًا؛ الأذنان للخلف، الذيل السريع، أو التحديق الحاد علامات تدل على الانزعاج. عندما تلاحظ هذه العلامات، تجنب لمس القطة حتى لا تشعر بالتهديد.",
        
        "اللعب التفاعلي اليومي يساعد كثيرًا في تقليل التوتر والطاقة السلبية عند القطط الغاضبة. استخدم ألعاب الصيد أو الكرات الخفيفة لتفريغ الطاقة بطريقة صحية وآمنة.",
        
        "تأكد من أن القطة تحصل على احتياجاتها الأساسية بانتظام مثل الطعام والماء ومكان النوم النظيف، لأن الإهمال أو التغيير المفاجئ قد يؤدي إلى زيادة السلوك العدواني.",
        
        "إذا استمر السلوك الغاضب لفترة طويلة أو أصبح شديدًا بشكل غير معتاد، فقد يكون السبب ألمًا أو مشكلة صحية. في هذه الحالة من الأفضل استشارة طبيب بيطري للتأكد من سلامتها."
    ],

    "relaxed": [
        "تبدو القطة مرتاحة وهادئة، وهذا يعني أنها تشعر بالأمان في البيئة المحيطة. حافظ على الروتين اليومي الهادئ الذي اعتادت عليه حتى تستمر بهذا الشعور الإيجابي.",
        
        "القطط الهادئة تستمتع غالبًا بالأماكن الدافئة والمريحة. وفر لها مساحة خاصة تحتوي على بطانية ناعمة أو سرير مريح بعيدًا عن الإزعاج والضوضاء.",
        
        "استغل هذه الحالة الهادئة لبناء علاقة قوية مع قطتك من خلال اللعب الخفيف أو المداعبة اللطيفة، فالحيوانات تكون أكثر تقبلًا للتفاعل عندما تكون مرتاحة.",
        
        "القطط المسترخية تحتاج أيضًا إلى تحفيز ذهني بسيط حتى لا تشعر بالملل. يمكنك استخدام ألعاب ذكية أو تغيير أماكن اللعب بين فترة وأخرى للحفاظ على نشاطها.",
        
        "من العلامات الجيدة للاسترخاء أن تكون وضعية الجسم طبيعية والأذنان للأمام أو بشكل مريح. استمر في توفير بيئة مستقرة لأن الاستقرار مهم جدًا لصحة القطط النفسية.",
        
        "احرص على مراقبة النظام الغذائي والنوم، فالقطط الهادئة غالبًا تكون بصحة جيدة. الحفاظ على روتين ثابت للطعام والراحة يساعد في استمرار هذا السلوك الإيجابي."
    ],

    "sad": [
        "إذا بدت القطة حزينة أو خاملة، حاول قضاء وقت أطول معها بلطف وهدوء. القطط قد تشعر بالوحدة أو التوتر خاصة عند تغير البيئة أو غياب أصحابها لفترات طويلة.",
        
        "وفّر للقطة مكانًا دافئًا وآمنًا تستطيع الاختباء أو الراحة فيه، لأن القطط الحزينة تميل أحيانًا للعزلة وتفضّل الأماكن الهادئة البعيدة عن الإزعاج.",
        
        "اللعب الخفيف والتحفيز التدريجي يمكن أن يساعد في تحسين الحالة النفسية للقطة. استخدم ألعابًا بسيطة أو قدم مكافآت صغيرة لتشجيعها على التفاعل.",
        
        "راقب الشهية والنشاط العام؛ إذا لاحظت انخفاضًا واضحًا في الأكل أو النوم الزائد لفترة طويلة، فقد يكون الحزن مرتبطًا بمشكلة صحية تحتاج متابعة بيطرية.",
        
        "التغييرات المفاجئة مثل الانتقال إلى منزل جديد أو دخول حيوان آخر قد تؤثر على نفسية القطة. حاول جعل التغييرات تدريجية وامنحها وقتًا للتأقلم.",
        
        "التفاعل الهادئ والصوت اللطيف يساعدان القطط الحزينة على الشعور بالأمان. تجنب الصراخ أو العقاب لأن ذلك قد يزيد من التوتر والخوف لديها."
    ]
}

dog_tips = {

    "angry": [

        "يبدو الكلب في حالة غضب أو توتر واضح، لذلك من الأفضل عدم الاقتراب منه بشكل مفاجئ أو محاولة لمسه بالقوة. الكلاب عندما تشعر بالخوف أو التهديد قد تتصرف بعدوانية للدفاع عن نفسها، لذلك امنحه مساحة هادئة حتى يستعيد شعوره بالأمان تدريجيًا.",

        "قد تظهر على الكلب علامات الغضب مثل التحديق الحاد أو شد الجسم أو إصدار أصوات منخفضة ومتقطعة. هذه الإشارات تعني أن الكلب غير مرتاح حاليًا، ومن المهم تجنب إثارة توتره أكثر أو إجباره على التفاعل.",

        "الأصوات العالية أو البيئات المزدحمة أو التغييرات المفاجئة قد تزيد من عصبية الكلب وغضبه. حاول توفير مكان هادئ وآمن يحتوي على سريره وألعابه المفضلة حتى يشعر بالاستقرار والراحة.",

        "التعامل الهادئ والصبور مع الكلب الغاضب مهم جدًا. الصراخ أو العقاب الجسدي قد يزيد من خوفه وعدائيته، بينما الصوت الهادئ والتصرفات البطيئة تساعده على الاسترخاء واستعادة ثقته.",

        "اللعب المنتظم والمشي اليومي يساعدان الكلاب على تفريغ الطاقة السلبية والتوتر، مما يقلل من احتمالية ظهور السلوك العدواني بشكل متكرر ويحسن حالتها النفسية بشكل عام.",

        "إذا لاحظت أن الغضب أو العدائية تتكرر بشكل مستمر أو مبالغ فيه، فقد يكون السبب مشكلة صحية أو ألم جسدي أو ضغط نفسي، لذلك من الأفضل استشارة طبيب بيطري أو مدرب مختص بسلوك الكلاب."
    ],

    "happy": [

        "يبدو الكلب سعيدًا ومرتاحًا، وهذه علامة جيدة على أنه يشعر بالأمان والحب في البيئة المحيطة به. الحفاظ على هذا الشعور يتطلب استمرار التفاعل الإيجابي والاهتمام اليومي به.",

        "الكلاب السعيدة تستمتع عادة باللعب والحركة والتفاعل مع أصحابها. تخصيص وقت يومي للمشي أو اللعب يساعد في الحفاظ على نشاط الكلب وتحسين حالته النفسية والجسدية.",

        "علامات السعادة مثل حركة الذيل بشكل طبيعي ووضعية الجسم المريحة تدل على أن الكلب يشعر بالثقة والارتياح. البيئة المستقرة والداعمة تساعده على الاستمرار بهذه الحالة الإيجابية.",

        "التعزيز الإيجابي مثل المكافآت والكلمات اللطيفة والمداعبة يساعد الكلب على بناء ارتباط قوي مع صاحبه ويزيد شعوره بالأمان والسعادة.",

        "الحفاظ على نظام غذائي جيد ونشاط بدني منتظم ونوم مريح يساعد الكلب على البقاء بصحة جيدة ومزاج مستقر، لأن الراحة الجسدية تؤثر بشكل مباشر على حالته النفسية.",

        "الكلاب السعيدة تكون عادة أكثر تقبلًا للتدريب والتعلم والتفاعل الاجتماعي، لذلك يمكن استغلال هذه الحالة لتعليمها أوامر جديدة أو تعزيز السلوكيات الإيجابية."
    ],

    "relaxed": [

        "يبدو الكلب هادئًا ومستقرًا نفسيًا، وهذا مؤشر جيد على أنه يشعر بالأمان والراحة في البيئة المحيطة به. الكلاب الهادئة غالبًا تكون في حالة توازن نفسي جيدة.",

        "الكلاب المسترخية تستفيد من الروتين الثابت والبيئة الهادئة، لذلك من الأفضل الحفاظ على أوقات منتظمة للطعام والنوم والمشي حتى يبقى الكلب مرتاحًا نفسيًا.",

        "توفير مكان مريح للنوم والراحة يساعد الكلب على الاستمرار في الشعور بالهدوء والاستقرار، خاصة إذا كان المكان بعيدًا عن الضوضاء أو الحركة المزعجة.",

        "حتى عندما يكون الكلب هادئًا، من المهم توفير بعض النشاط البدني والتحفيز الذهني البسيط مثل الألعاب أو التمارين الخفيفة للحفاظ على توازنه النفسي.",

        "التفاعل اللطيف والمداعبة الهادئة يساعدان في تعزيز شعور الكلب بالأمان ويقويان العلاقة بينه وبين صاحبه، مما ينعكس إيجابيًا على سلوكه العام.",

        "البيئة المستقرة والخالية من التوتر تساعد الكلاب على البقاء هادئة نفسيًا وتقلل من احتمالية ظهور القلق أو العصبية أو السلوك العدواني."
    ],

    "sad": [

        "قد يبدو الكلب حزينًا أو منخفض النشاط، وربما يحتاج إلى اهتمام إضافي أو وقت أطول مع صاحبه ليشعر بالراحة والأمان. التفاعل الهادئ قد يساعده على تحسين حالته النفسية تدريجيًا.",

        "الكلاب الحزينة قد تميل إلى قلة الحركة أو فقدان الاهتمام باللعب والطعام، لذلك من المهم مراقبة التغيرات السلوكية بشكل مستمر والانتباه لأي علامات غير طبيعية.",

        "حاول تشجيع الكلب على اللعب أو المشي بلطف، لأن النشاط البدني والتفاعل الإيجابي يساعدان في تحسين مزاجه وتقليل الشعور بالحزن أو الخمول.",

        "التغييرات المفاجئة مثل الانتقال إلى مكان جديد أو غياب شخص اعتاد عليه الكلب قد تؤثر بشكل كبير على حالته النفسية وتجعله أكثر انعزالًا أو توترًا.",

        "التعامل بصوت هادئ وإظهار الاهتمام والرعاية يساعد الكلب الحزين على استعادة ثقته والشعور بالراحة تدريجيًا، خاصة إذا كان يمر بفترة توتر أو تغيير.",

        "إذا استمرت علامات الحزن أو الخمول لفترة طويلة أو لاحظت تغيرًا واضحًا في الشهية أو النوم أو النشاط، فمن الأفضل استشارة طبيب بيطري للتأكد من عدم وجود مشكلة صحية."
    ]
}

# =========================================
# DOG YOUTUBE QUERIES
# =========================================

dog_youtube_queries_ar = {

    "angry": [
        "السلوك العدواني للكلاب والهيبره وكيفيه علاجها",
        "افهم لغة جسد الكلاب في الحالات المزاجية مثل متى يكون سعيد او خايف او قلق ومتوتر او غضبان",
        "١٠ تصرفات تغضب الكلاب منك / الجزء الأول",
        "لحظة هجوم الكلاب علي أصحابها وما هي الاسباب | ثلاث أسباب لهجوم الكلاب وكيفية تجنبها",
        "اذا فعلت هذا الخطأ كلبك يعضك بهجوم شرس وخطير"
    ],

    "happy": [
        "علامات سعادة كلبك: تعرف عليها الآن",
        "10 Signs Your Dog Is Truly Happy and Healthy – #7 Will Surprise You!",
        "طريقة اللعب الصحيحه مع كلبك في البيت 🐶",
        "علامات حب الكلاب لاصحابها | هل كلبك يحبك او لا",
    ],

    "relaxed": [
        "Dog Body Language 101",
        "Your Dog May Look Calm — (Here’s Whether They’re Relaxed, Alert, or Quietly Stressed)",
        "The Calm Settle - for dogs and puppies",
        "5 ways to build a better relationship with your dog",
    ],

    "sad": [
         "اعراض الاكتئاب عند الكلاب وكيف يمكنك ابتهاجه",
        "إكتئاب الكلاب..؟ 😳",
        "هل لكلبك مشاعر مثل الانسان . هل يحزن الكلب. يكتئب. اسباب حزن الكلب وكيف نعالجها ؟",
        "لغة الكلاب / لغة الجسد للكلاب / اسرار لغة الكلاب / افهم كلبك بيقول اي / سامر غازي",
        "٥ أسباب للخمول عند الكلاب مع الدكتور هاشم طبيب بيطري"
    ]
}

# =========================================
# CAT YOUTUBE QUERIES
# =========================================

cat_youtube_queries_ar = {

    "angry": [
        "كيف تتعامل مع القطط الشرسة و العنيدة مع الدكتور رامي 😱 جزء 1",
        "كيف تعرف أن قطتك غاضبة منك؟",
        "خمسة علامات تدل على غضب القطط من الانسان🐈😠 #قطط",
        "سبب هجوم القطط على أصحابها",
        "8 Types of Cat Aggression Explained!"
    ],

    "relaxed": [
        "علامات راحة القطط",
        "كيف أعرف أن قطتي مرتاحة",
        "سلوك القطط الهادئة",
        "لغة جسد القطط المسترخية",
         "Instantly Improve Your Cat's Life with these 7 Things",
        "هل قطك سعيد معك؟ 🔍 اكتشف 4 علامات تؤكد ذلك! #قطط #القطط"
    ],

    "sad": [
        "5 علامات تدل على زعل القطط من الإنسان 🐈‍⬛🐈 #قطط #قطة",
        "10 علامات تدل على أن قطك حزين جدا",
        "اكتئاب القطط",
        "اكتئاب القطط / الأسباب ؛ الأعراض والعلاج ؟؟",
        "5 أشياء تجرح مشاعر القطط 🐈‍⬛😿 #قطط #قطة",
        "هذه 5 أسباب تجعل قطك حزين"
    ]
}


# =========================================
# GET VIDEO
# =========================================

def get_random_arabic_youtube_video(emotion, animal):

    if animal == "dog":
        queries = dog_youtube_queries_ar

    elif animal == "cat":
        queries = cat_youtube_queries_ar

    else:
        return None

    query = random.choice(
        queries.get(emotion, ["الحيوانات"])
    )

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoEmbeddable": "true",
        "maxResults": 15,
        "relevanceLanguage": "ar",
        "regionCode": "SA",
        "safeSearch": "strict",
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(url, params=params)

    data = response.json()

    if "items" not in data or len(data["items"]) == 0:
        return None

    random_video = random.choice(data["items"])

    video_id = random_video["id"]["videoId"]

    return {
        "title": random_video["snippet"]["title"],
        "channel": random_video["snippet"]["channelTitle"],
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": random_video["snippet"]["thumbnails"]["medium"]["url"]
    }

# =========================================
# IMAGE PREPROCESS
# =========================================

def prepare_image(file_bytes):

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array

# =========================================
# MAIN API
# =========================================
@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...)
):
    file_bytes = await file.read()

    detected = detect_animal_yolo(file_bytes)

    if detected is None:
        return {
            "is_animal_supported": False,
            "message": "الصورة لا تحتوي على كلب أو قطة بوضوح."
        }

    animal = detected["animal"]
    detected_confidence = detected["confidence"]

    img_array = prepare_image(file_bytes)

    if animal == "dog":
        prediction = get_dog_model().predict(img_array)[0]

        pred_index = int(np.argmax(prediction))
        emotion = dog_class_labels[pred_index]
        confidence = float(prediction[pred_index])

        return {
            "is_animal_supported": True,
            "animal": "dog",
            "animal_detection_confidence": detected_confidence,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                dog_class_labels[i]: float(prediction[i])
                for i in range(len(dog_class_labels))
            },
            "tip": random.choice(dog_tips[emotion]),
            "video": get_random_arabic_youtube_video(emotion, "dog")
        }

    if animal == "cat":
        prediction = get_cat_model().predict(img_array)[0]
        
        emotion, confidence, adjusted_prediction = apply_cat_behavior_rules(
            prediction,
            cat_class_labels
        )

        return {
            "is_animal_supported": True,
            "animal": "cat",
            "animal_detection_confidence": detected_confidence,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                cat_class_labels[i]: float(adjusted_prediction[i])
                for i in range(len(cat_class_labels))
            },
            "tip": random.choice(cat_tips[emotion]),
            "video": get_random_arabic_youtube_video(emotion, "cat")
        }
    
