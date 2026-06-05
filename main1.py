import os
import io
import json
import random
import re
import requests
import time
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
YOUTUBE_CACHE_TTL_SECONDS = 60 * 60 * 12

# =========================================
# DOWNLOAD DOG MODEL
# =========================================

MODEL_DIR = os.getenv("MODEL_DIR", ".")
os.makedirs(MODEL_DIR, exist_ok=True)
YOUTUBE_CACHE_PATH = os.path.join(MODEL_DIR, "youtube_video_cache.json")

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
    "sad",
    "angry",
    "relaxed"
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

    top_prob = max(probs.values())

    if top_prob >= 0.80:
        adjusted = prediction

    else:
        for i, cls in enumerate(class_labels):

            if cls == "angry":
                adjusted[i] *= 1.00

            elif cls == "relaxed":
                adjusted[i] *= 0.95

            elif cls == "sad":
                adjusted[i] *= 1.05

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

dog_youtube_queries_en = {
    "angry": [
        "aggressive dog behavior how to handle",
        "angry dog body language signs",
        "how to calm an aggressive dog",
        "dog aggression causes and training",
        "how to deal with a reactive dog"
    ],
    "happy": [
        "signs your dog is happy",
        "how to know your dog is happy",
        "happy dog body language",
        "how to make your dog happy and healthy",
        "ways dogs show love to owners"
    ],
    "relaxed": [
        "relaxed dog body language",
        "signs your dog is calm and relaxed",
        "how to calm your dog",
        "dog relaxation training",
        "how to teach a dog to settle"
    ],
    "sad": [
        "signs your dog is sad",
        "dog depression symptoms",
        "how to help a sad dog",
        "why is my dog sad",
        "sad dog body language"
    ]
}

cat_youtube_queries_en = {
    "angry": [
        "angry cat body language",
        "signs your cat is angry",
        "how to calm an angry cat",
        "cat aggression causes and solutions",
        "why cats attack owners"
    ],
    "relaxed": [
        "relaxed cat body language",
        "signs your cat is relaxed",
        "how to calm your cat",
        "signs your cat is happy and comfortable",
        "cat body language relaxed"
    ],
    "sad": [
        "signs your cat is sad",
        "cat depression symptoms",
        "how to help a sad cat",
        "why is my cat sad",
        "sad cat body language"
    ]
}

# =========================================
# GET VIDEO
# =========================================

youtube_filter_words = {
    "dog": {
        "animal": ["كلب", "كلاب", "كلبك", "الكلب", "dog", "dogs", "puppy"],
        "angry": ["غاضب", "غضب", "عدواني", "عدوانية", "هجوم", "عض", "متوتر", "شرس", "تهدئة", "ينبح", "تنبح", "نباح", "aggressive", "aggression", "angry", "reactive", "bite", "barking", "calm"],
        "happy": ["سعيد", "سعادة", "فرح", "يلعب", "اللعب", "حب", "يحبك", "happy", "happiness", "love", "play", "joy"],
        "relaxed": ["هادئ", "هدوء", "استرخاء", "مسترخي", "راحة", "لغة جسد", "طاعة", "مطيع", "relaxed", "calm", "settle", "relaxation", "body language"],
        "sad": ["حزين", "حزن", "اكتئاب", "خمول", "منخفض", "زعلان", "مكتئب", "sad", "depressed", "depression", "unhappy", "body language"]
    },
    "cat": {
        "animal": ["قطة", "قطط", "قطتك", "القطة", "قط", "cat", "cats", "kitten"],
        "angry": ["غاضبة", "غاضب", "غضب", "عدوانية", "هجوم", "شرسة", "عض", "تهدئة", "لا تحبك", "تصرفات", "angry", "aggressive", "aggression", "attack", "calm"],
        "relaxed": ["هادئة", "هادئ", "هدوء", "استرخاء", "مسترخية", "راحة", "مرتاحة", "سعيدة", "تهدئ", "relaxed", "calm", "comfortable", "happy", "body language"],
        "sad": ["حزينة", "حزين", "حزن", "اكتئاب", "زعلانة", "خمول", "مكتئبة", "sad", "depressed", "depression", "unhappy", "body language"]
    }
}

youtube_emotion_excluded_words = {
    "relaxed": ["ينبح", "تنبح", "نباح", "هجوم", "عدوان", "غاضب", "غضب"],
    "happy": ["حزين", "حزن", "اكتئاب", "غاضب", "غضب", "هجوم", "sad", "depression", "aggressive"],
    "sad": ["الكلب الاسود", "الكلب الأسود", "كلب اسود", "كلب أسود", "كتاب", "مريض", "مرض", "التعافي", "البشر", "الانسان", "الإنسان", "ببساطه", "ببساطة", "black dog", "book", "human depression"],
    "angry": ["سعيد", "سعادة", "هادئ", "هدوء", "استرخاء", "happy", "relaxed"]
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

youtube_fallback_videos = {
    "dog": {
        "angry": [
            {
                "title": "لحظة هجوم الكلاب علي أصحابها وما هي الاسباب | ثلاث أسباب لهجوم الكلاب وكيفية تجنبها",
                "channel": "Mafia Dogs & Cats عصابة الكلاب و القطط",
                "url": "https://www.youtube.com/watch?v=HWJEpZ1DYjU",
                "thumbnail": "https://i.ytimg.com/vi/HWJEpZ1DYjU/mqdefault.jpg"
            }
        ],
        "happy": [
            {
                "title": "علامات سعادة كلبك: تعرف عليها الآن",
                "channel": "حيواناتي",
                "url": "https://www.youtube.com/watch?v=Grv1Vq2_-H4",
                "thumbnail": "https://i.ytimg.com/vi/Grv1Vq2_-H4/mqdefault.jpg"
            }
        ],
        "relaxed": [
            {
                "title": "تدريب الكلب علي الطاعة تدريب الكلب علي اداب الطعام التدريب علي امر المنع مع سامر تي في دوج",
                "channel": "تي في دوج Tv Dogs",
                "url": "https://www.youtube.com/watch?v=Cn9rcMzgdJs",
                "thumbnail": "https://i.ytimg.com/vi/Cn9rcMzgdJs/mqdefault.jpg"
            }
        ],
        "sad": [
            {
                "title": "هل كلبك حزين؟ علامات لا ينتبه لها معظم الناس",
                "channel": "لغة الوفاء",
                "url": "https://www.youtube.com/watch?v=fcR8vRLfSes",
                "thumbnail": "https://i.ytimg.com/vi/fcR8vRLfSes/mqdefault.jpg"
            }
        ]
    },
    "cat": {
        "angry": [
            {
                "title": "سبب هجوم القطط على أصحابها",
                "channel": "Dr Wahid - دكتور مياو",
                "url": "https://www.youtube.com/watch?v=clD1uFzWhBA",
                "thumbnail": "https://i.ytimg.com/vi/clD1uFzWhBA/mqdefault.jpg"
            }
        ],
        "relaxed": [
            {
                "title": "فهم لغة جسد القطط: كيف تعرف إذا كانت قطتك سعيدة أو متوترة؟",
                "channel": "Vet-C",
                "url": "https://www.youtube.com/watch?v=mIMIhgZLGHQ",
                "thumbnail": "https://i.ytimg.com/vi/mIMIhgZLGHQ/mqdefault.jpg"
            }
        ],
        "sad": [
            {
                "title": "نصائح لعلاج اكتئاب القطط",
                "channel": "Farah samir",
                "url": "https://www.youtube.com/watch?v=NC0qivFiNZo",
                "thumbnail": "https://i.ytimg.com/vi/NC0qivFiNZo/mqdefault.jpg"
            }
        ]
    }
}


def _contains_any(text, words):
    normalized = text.lower()
    return any(word.lower() in normalized for word in words)


def _has_arabic_text(text):
    return re.search(r"[\u0600-\u06FF]", text) is not None


def _is_relevant_youtube_item(item, animal, emotion, require_arabic):
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

    if require_arabic and not _has_arabic_text(title):
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

def _load_youtube_cache():
    try:
        with open(YOUTUBE_CACHE_PATH, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_youtube_cache(cache):
    try:
        with open(YOUTUBE_CACHE_PATH, "w", encoding="utf-8") as cache_file:
            json.dump(cache, cache_file, ensure_ascii=False)
    except OSError:
        pass


def _get_cached_youtube_videos(cache_key):
    cache = _load_youtube_cache()
    entry = cache.get(cache_key)

    if not entry:
        return None

    if time.time() - entry.get("created_at", 0) > YOUTUBE_CACHE_TTL_SECONDS:
        return None

    videos = entry.get("videos", [])
    return videos or None


def _set_cached_youtube_videos(cache_key, videos):
    cache = _load_youtube_cache()
    cache[cache_key] = {
        "created_at": time.time(),
        "videos": videos
    }
    _save_youtube_cache(cache)


def _search_youtube_videos(queries, animal, emotion, require_arabic, relevance_language, region_code):
    url = "https://www.googleapis.com/youtube/v3/search"
    candidate_videos = []

    selected_queries = queries.get(emotion, []).copy()
    random.shuffle(selected_queries)

    for query in selected_queries[:1]:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "videoEmbeddable": "true",
            "videoDuration": "medium",
            "maxResults": 25,
            "relevanceLanguage": relevance_language,
            "regionCode": region_code,
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
            if _is_relevant_youtube_item(item, animal, emotion, require_arabic):
                candidate_videos.append(_build_youtube_video(item))

        if candidate_videos:
            break

    return candidate_videos


def get_random_arabic_youtube_video(emotion, animal):

    if not YOUTUBE_API_KEY:
        return None

    if animal == "dog":
        queries = dog_youtube_queries_ar
        fallback_queries = dog_youtube_queries_en

    elif animal == "cat":
        queries = cat_youtube_queries_ar
        fallback_queries = cat_youtube_queries_en

    else:
        return None

    cache_key = f"{animal}:{emotion}:youtube_recommendations"
    cached_videos = _get_cached_youtube_videos(cache_key)

    if cached_videos:
        return random.choice(cached_videos)

    arabic_videos = _search_youtube_videos(
        queries,
        animal,
        emotion,
        require_arabic=True,
        relevance_language="ar",
        region_code="SA"
    )

    if arabic_videos:
        _set_cached_youtube_videos(cache_key, arabic_videos)
        return random.choice(arabic_videos)

    english_videos = _search_youtube_videos(
        fallback_queries,
        animal,
        emotion,
        require_arabic=False,
        relevance_language="en",
        region_code="US"
    )

    if english_videos:
        _set_cached_youtube_videos(cache_key, english_videos)
        return random.choice(english_videos)

    fallback_videos = youtube_fallback_videos.get(animal, {}).get(emotion, [])
    if fallback_videos:
        return random.choice(fallback_videos)

    return {
        "title": "فيديو تعليمي عن رعاية الحيوانات الأليفة",
        "channel": "YouTube",
        "url": "https://www.youtube.com/results?search_query=pet+care+animal+behavior",
        "thumbnail": "https://i.ytimg.com/vi/Grv1Vq2_-H4/mqdefault.jpg"
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
    
    

