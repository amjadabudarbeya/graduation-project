import os
import io
import json
import random
import re
import requests
import gdown
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense

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

MODEL_DIR = os.getenv("MODEL_DIR", ".")
os.makedirs(MODEL_DIR, exist_ok=True)

DOG_MODEL_PATH = os.path.join(MODEL_DIR, "dog_model.keras")

if not os.path.exists(DOG_MODEL_PATH):

    gdown.download(
        url="https://drive.google.com/file/d/15s4lneWlkWg_Acf2NE5szuIZkBXnR3bl/view?usp=drive_link",
        output=DOG_MODEL_PATH,
        quiet=False
    )

# =========================================
# DOWNLOAD CAT MODEL
# =========================================

CAT_MODEL_PATH = os.path.join(MODEL_DIR, "cat_model.keras")

if not os.path.exists(CAT_MODEL_PATH):

    gdown.download(
        url="https://drive.google.com/file/d/1IauPJI2NbPSwlQ2giJO3z3nqtHI33ifh/view?usp=sharing",
        output=CAT_MODEL_PATH,
        quiet=False
    )

# =========================================
# LOAD MODELS
# =========================================

def _patch_dense_config_for_saved_models():
    original_from_config = Dense.from_config

    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_from_config(config)

    Dense.from_config = classmethod(from_config)


_patch_dense_config_for_saved_models()

dog_model = tf.keras.models.load_model(DOG_MODEL_PATH, compile=False)
cat_model = tf.keras.models.load_model(CAT_MODEL_PATH, compile=False)

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

        "يبدو الكلب في حالة غضب أو توتر واضح، لذلك من الأفضل عدم الاقتراب منه بشكل مفاجئ أو محاولة لمسه بالقوة. امنحه مساحة هادئة حتى يشعر بالأمان ويهدأ تدريجيًا.",

        "عندما يكون الكلب غاضبًا قد تظهر علامات مثل التحديق الحاد أو إصدار أصوات منخفضة أو شد الجسم بشكل واضح. هذه الإشارات تعني أنه يشعر بالتهديد أو الانزعاج ويحتاج للهدوء.",

        "حاول تقليل الضوضاء أو أي مصدر يسبب التوتر للكلب، لأن الأصوات المرتفعة أو البيئات المزدحمة قد تزيد من السلوك العدواني أو العصبي.",

        "الكلاب الغاضبة تحتاج إلى تعامل هادئ وصبور. تجنب الصراخ أو العقاب الجسدي لأن ذلك قد يزيد خوف الكلب وعدائيته بدلًا من تهدئته.",

        "التمارين اليومية واللعب التفاعلي يساعدان الكلاب على التخلص من الطاقة السلبية والتوتر، مما يقلل من احتمالية ظهور السلوك العدواني بشكل متكرر.",

        "إذا لاحظت أن الغضب أو العدائية تتكرر بشكل مستمر أو مبالغ فيه، فقد يكون السبب مشكلة صحية أو نفسية، لذلك يُفضل استشارة طبيب بيطري أو مدرب مختص."
    ],

    "happy": [

        "يبدو الكلب سعيدًا ومرتاحًا، وهذا يدل غالبًا على أنه يشعر بالأمان والحب في البيئة المحيطة به. حافظ على التفاعل الإيجابي والروتين الذي يجعله مرتاحًا.",

        "الكلاب السعيدة تستمتع باللعب والحركة والتفاعل مع أصحابها. تخصيص وقت يومي للعب أو المشي يساعد في الحفاظ على حالته النفسية الجيدة.",

        "علامات السعادة مثل الذيل المتحرك ووضعية الجسم المريحة تدل على أن الكلب يشعر بالثقة والراحة. استمر في توفير بيئة مستقرة وآمنة له.",

        "التعزيز الإيجابي مثل المكافآت والكلمات اللطيفة يساعد الكلب على بناء ارتباط إيجابي مع صاحبه ويزيد شعوره بالسعادة والراحة.",

        "الحفاظ على نظام غذائي جيد ونشاط بدني منتظم يساعد الكلب على البقاء نشيطًا وسعيدًا وصحيًا على المدى الطويل.",

        "الكلاب السعيدة تكون عادة أكثر تقبلًا للتدريب والتفاعل الاجتماعي، لذلك يمكن استغلال هذه الحالة لتعليمها أوامر جديدة بطريقة ممتعة."
    ],

    "relaxed": [

        "يبدو الكلب هادئًا ومستقرًا نفسيًا، وهذه علامة جيدة على أنه يشعر بالأمان والراحة في البيئة المحيطة به.",

        "الكلاب المسترخية تستفيد من الروتين الهادئ والمستقر، لذلك حاول الحفاظ على أوقات ثابتة للطعام والنوم والمشي.",

        "توفير مكان مريح للنوم والراحة يساعد الكلب على الاستمرار في الشعور بالهدوء والاستقرار النفسي.",

        "حتى عندما يكون الكلب هادئًا، من المهم توفير نشاط بدني وتحفيز ذهني بسيط للحفاظ على توازنه النفسي والجسدي.",

        "التفاعل اللطيف والمداعبة الهادئة يساعدان في تعزيز شعور الكلب بالأمان وتقوية العلاقة بينه وبين صاحبه.",

        "البيئة الهادئة والخالية من التوتر تساعد الكلاب على البقاء مستقرة نفسيًا وتقلل من احتمالية ظهور القلق أو العصبية."
    ],

    "sad": [

        "قد يبدو الكلب حزينًا أو منخفض النشاط، وربما يحتاج إلى اهتمام إضافي أو وقت أطول مع صاحبه ليشعر بالراحة والأمان.",

        "الكلاب الحزينة قد تميل إلى قلة الحركة أو فقدان الاهتمام باللعب والطعام، لذلك من المهم مراقبة التغيرات السلوكية بشكل مستمر.",

        "حاول تشجيع الكلب على اللعب أو المشي بلطف، لأن النشاط البدني والتفاعل الإيجابي يساعدان في تحسين حالته النفسية.",

        "التغييرات المفاجئة مثل الانتقال إلى مكان جديد أو غياب شخص اعتاد عليه الكلب قد تؤثر على مزاجه وتسبب له التوتر أو الحزن.",

        "التعامل بصوت هادئ وإظهار الاهتمام والرعاية يساعد الكلب الحزين على استعادة ثقته والشعور بالراحة تدريجيًا.",

        "إذا استمرت علامات الحزن أو الخمول لفترة طويلة أو لاحظت تغيرًا واضحًا في الشهية أو النوم، فمن الأفضل استشارة طبيب بيطري للتأكد من عدم وجود مشكلة صحية."
    ]
}

# =========================================
# DOG YOUTUBE QUERIES
# =========================================

dog_youtube_queries_ar = {

    "angry": [
        "كلب غاضب كيف تتعامل معه",
        "السلوك العدواني عند الكلاب وطرق التعامل معه",
        "علامات غضب الكلب ولغة الجسد",
        "تهدئة الكلب الغاضب أو المتوتر",
        "أسباب عدوانية الكلاب وكيفية علاجها"
    ],

    "happy": [
        "علامات سعادة الكلب",
        "كيف تعرف أن كلبك سعيد",
        "اللعب الصحيح مع الكلاب السعيدة",
        "علامات حب الكلب لصاحبه",
        "كيف تجعل كلبك سعيد وصحي"
    ],

    "relaxed": [
        "علامات هدوء الكلب واسترخائه",
        "لغة جسد الكلب الهادئ",
        "كيف تتعامل مع كلب هادئ",
        "تدريب الكلب على الهدوء والاسترخاء",
        "روتين يساعد الكلب على الهدوء",
        "كيف تجعل كلبك هادئ",
        "تدريب الكلب على الهدوء والطاعة"
    ],

    "sad": [
        "علامات حزن الكلب",
        "اكتئاب الكلاب وأسبابه وعلاجه",
        "كيف تساعد كلب حزين",
        "أسباب خمول وحزن الكلاب",
        "لغة جسد الكلب الحزين"
    ]
}

# =========================================
# CAT YOUTUBE QUERIES
# =========================================

cat_youtube_queries_ar = {

    "angry": [
        "قطة غاضبة كيف تتعامل معها",
        "علامات غضب القطط",
        "السلوك العدواني عند القطط",
        "تهدئة القطة الغاضبة",
        "أسباب هجوم القطط على أصحابها"
    ],

    "relaxed": [
        "علامات راحة القطط",
        "كيف أعرف أن قطتي مرتاحة",
        "سلوك القطط الهادئة",
        "لغة جسد القطة المسترخية",
        "كيف تجعل القطة هادئة ومرتاحة",
        "علامات القطة السعيدة والمرتاحة",
        "كيف تهدئ قطتك"
    ],

    "sad": [
        "علامات حزن القطط",
        "اكتئاب القطط الأسباب والعلاج",
        "كيف تساعد قطة حزينة",
        "أسباب حزن القطة",
        "لغة جسد القطة الحزينة",
        "كيف أعرف أن قطتي حزينة",
        "علاج حزن القطط"
    ]
}

# =========================================
# GET VIDEO
# =========================================

youtube_filter_words = {
    "dog": {
        "animal": ["كلب", "كلاب", "كلبك", "الكلب"],
        "angry": ["غاضب", "غضب", "عدواني", "عدوانية", "هجوم", "عض", "متوتر", "شرس", "تهدئة", "ينبح", "تنبح", "نباح"],
        "happy": ["سعيد", "سعادة", "فرح", "يلعب", "اللعب", "حب", "يحبك"],
        "relaxed": ["هادئ", "هدوء", "استرخاء", "مسترخي", "راحة", "لغة جسد", "طاعة", "مطيع"],
        "sad": ["حزين", "حزن", "اكتئاب", "خمول", "منخفض", "زعلان", "مكتئب"]
    },
    "cat": {
        "animal": ["قطة", "قطط", "قطتك", "القطة", "قط"],
        "angry": ["غاضبة", "غاضب", "غضب", "عدوانية", "هجوم", "شرسة", "عض", "تهدئة", "لا تحبك", "تصرفات"],
        "relaxed": ["هادئة", "هادئ", "هدوء", "استرخاء", "مسترخية", "راحة", "مرتاحة", "سعيدة", "تهدئ"],
        "sad": ["حزينة", "حزين", "حزن", "اكتئاب", "زعلانة", "خمول", "مكتئبة"]
    }
}

youtube_emotion_excluded_words = {
    "relaxed": ["ينبح", "تنبح", "نباح", "هجوم", "عدوان", "غاضب", "غضب"],
    "happy": ["حزين", "حزن", "اكتئاب", "غاضب", "غضب", "هجوم"],
    "sad": ["الكلب الاسود", "الكلب الأسود", "كلب اسود", "كلب أسود", "كتاب", "مريض", "مرض", "التعافي", "البشر", "الانسان", "الإنسان", "ببساطه", "ببساطة"],
    "angry": ["سعيد", "سعادة", "هادئ", "هدوء", "استرخاء"]
}

youtube_excluded_words = [
    "اغنية",
    "أغنية",
    "كليب",
    "موسيقى",
    "رقص",
    "ضحك",
    "مضحك",
    "طريف",
    "طريفة",
    "طرائف",
    "ميمز",
    "تيك توك",
    "tiktok",
    "shorts",
    "cartoon",
    "كرتون",
    "انمي",
    "لعبة",
    "game",
    "movie",
    "film"
]


def _contains_any(text, words):
    normalized = text.lower()
    return any(word.lower() in normalized for word in words)


def _has_arabic_text(text):
    return re.search(r"[\u0600-\u06FF]", text) is not None


def _is_relevant_youtube_item(item, animal, emotion):
    snippet = item.get("snippet", {})
    title = snippet.get("title", "")
    description = snippet.get("description", "")
    searchable_text = " ".join([
        title,
        description
    ])

    filters = youtube_filter_words.get(animal, {})
    animal_words = filters.get("animal", [])
    emotion_words = filters.get(emotion, [])

    if not _has_arabic_text(title):
        return False

    if _contains_any(searchable_text, youtube_excluded_words):
        return False

    if _contains_any(searchable_text, youtube_emotion_excluded_words.get(emotion, [])):
        return False

    return (
        _contains_any(title, animal_words)
        and _contains_any(title, emotion_words)
    )


def _build_youtube_video(item):
    snippet = item["snippet"]
    video_id = item["id"]["videoId"]
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = (
        thumbnails.get("medium", {})
        or thumbnails.get("high", {})
        or thumbnails.get("default", {})
    ).get("url")

    return {
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail": thumbnail
    }

def get_random_arabic_youtube_video(emotion, animal):

    if not YOUTUBE_API_KEY:
        return None

    if animal == "dog":
        queries = dog_youtube_queries_ar

    elif animal == "cat":
        queries = cat_youtube_queries_ar

    else:
        return None

    url = "https://www.googleapis.com/youtube/v3/search"
    candidate_videos = []

    selected_queries = queries.get(emotion, []).copy()
    random.shuffle(selected_queries)

    for query in selected_queries[:3]:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoEmbeddable": "true",
            "videoDuration": "medium",
            "maxResults": 25,
            "relevanceLanguage": "ar",
            "regionCode": "SA",
            "safeSearch": "strict",
            "key": YOUTUBE_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
        except requests.RequestException:
            continue

        data = response.json()
        items = data.get("items", [])

        for item in items:
            if _is_relevant_youtube_item(item, animal, emotion):
                candidate_videos.append(item)

        if candidate_videos:
            break

    if not candidate_videos:
        return None

    return _build_youtube_video(random.choice(candidate_videos))

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

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    animal: str = Form(...)
):

    file_bytes = await file.read()

    img_array = prepare_image(file_bytes)

    # =====================================
    # DOG
    # =====================================

    if animal == "dog":

        prediction = dog_model.predict(img_array)[0]

        pred_index = int(np.argmax(prediction))

        emotion = dog_class_labels[pred_index]

        confidence = float(prediction[pred_index])

        return {

            "animal": "dog",

            "emotion": emotion,

            "confidence": confidence,

            "probabilities": {
                dog_class_labels[i]: float(prediction[i])
                for i in range(len(dog_class_labels))
            },

            "tip": random.choice(dog_tips[emotion]),

            "video": get_random_arabic_youtube_video(
                emotion,
                "dog"
            )
        }

    # =====================================
    # CAT
    # =====================================

    elif animal == "cat":

        prediction = cat_model.predict(img_array)[0]

        emotion, confidence, adjusted_prediction = apply_cat_behavior_rules(
            prediction,
            cat_class_labels
        )

        return {

            "animal": "cat",

            "emotion": emotion,

            "confidence": confidence,

            "probabilities": {
                cat_class_labels[i]: float(adjusted_prediction[i])
                for i in range(len(cat_class_labels))
            },

            "tip": random.choice(cat_tips[emotion]),

            "video": get_random_arabic_youtube_video(
                emotion,
                "cat"
            )
        }

    return {
        "error": "animal must be dog or cat"
    }
    

