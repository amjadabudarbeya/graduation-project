import modal


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]",
        "tensorflow-cpu",
        "numpy",
        "pillow",
        "python-multipart",
        "requests",
        "gdown",
    )
    .pip_install("keras-nightly")
    .add_local_file("main1.py", "/root/main1.py", copy=True)
    .add_local_file("cat_class_indices.json", "/root/cat_class_indices.json", copy=True)
    .add_local_file("dog_class_indices.json", "/root/dog_class_indices.json", copy=True)
)

models_volume = modal.Volume.from_name("petsense-emotion-models", create_if_missing=True)

app = modal.App("petsense-emotion-analysis")


@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=900,
)
def warmup_models():
    import os
    import sys

    os.environ["MODEL_DIR"] = "/models"
    sys.path.insert(0, "/root")

    import main1  # noqa: F401

    models_volume.commit()
    return "models are ready"


@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import os
    import sys

    os.environ["MODEL_DIR"] = "/models"
    sys.path.insert(0, "/root")

    from main1 import app as web_app

    return web_app
