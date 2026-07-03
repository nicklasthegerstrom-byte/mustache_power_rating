from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import io

from api.models import predict
from api.scoring import weighted_epic_score, compress_top_end, classify_epicness

register_heif_opener()

app = FastAPI()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Kunde inte läsa bilden.")

    result = predict(image)

    if result is None:
        raise HTTPException(status_code=422, detail="Inget ansikte hittades.")

    if not result["mustache"]:
        return {
            "score": 0,
            "title": "Mustaschlös",
            "mustache": False,
        }

    p_epic = result["p_epic"]
    p_medium = result["p_medium"]
    p_thin = result["p_thin"]

    epic_for_score, medium_for_score, thin_for_score = p_epic, p_medium, p_thin
    if p_epic >= 0.99:
        epic_for_score, medium_for_score, thin_for_score = 1.0, 0.0, 0.0

    score = weighted_epic_score(epic_for_score, medium_for_score, thin_for_score)
    score = compress_top_end(score)

    return {
        "score": round(score),
        "title": classify_epicness(score),
        "mustache": True,
        "p_epic": p_epic,
        "p_medium": p_medium,
        "p_thin": p_thin,
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
