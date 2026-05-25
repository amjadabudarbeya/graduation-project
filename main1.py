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

    if "items" not in data:
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
    

