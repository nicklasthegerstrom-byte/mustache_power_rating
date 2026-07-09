import asyncio
import base64
import io
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pillow_heif import register_heif_opener

from api.models import predict
from api.scoring import weighted_epic_score, compress_top_end, classify_epicness
from api import logger

register_heif_opener()

app = FastAPI()

# Certificate constants (match app.py)
PHOTO_X, PHOTO_Y, PHOTO_W, PHOTO_H = 428, 548, 1001, 1019
SCORE_Y, SCORE_H = 1745, 276
CLASS_Y, CLASS_H = 1620, 45
IMAGE_CENTER_X = 920
SHARE_FILL = (54, 88, 159)


def _cover_crop(img, w, h):
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


def _load_font(size):
    for path in ["assets/fonts/FalstaffMTStd.otf", "assets/fonts/Falstaff.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _strip_emoji(text):
    import unicodedata
    return "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk") and (c.isascii() or c.isalpha()))


def _draw_score_block(draw, score_str, cx, cy, score_font, suffix_font, fill):
    gap = 20
    sb = draw.textbbox((0, 0), score_str, font=score_font)
    score_w, score_h = sb[2] - sb[0], sb[3] - sb[1]
    fb = draw.textbbox((0, 0), "/100", font=suffix_font)
    suffix_w, suffix_h = fb[2] - fb[0], fb[3] - fb[1]
    total_w = score_w + gap + suffix_w
    start_x = cx - total_w // 2
    draw.text((start_x - sb[0], cy - score_h // 2 - sb[1]), score_str, font=score_font, fill=fill)
    suffix_x = start_x + score_w + gap
    suffix_y = cy + score_h // 2 - suffix_h - sb[1]
    draw.text((suffix_x - fb[0], suffix_y - fb[1]), "/100", font=suffix_font, fill=fill)


def _draw_class_text(draw, text, cx, cy, max_w, fill, max_size=76):
    font_path = "assets/fonts/Montserrat-Black.ttf"
    for size in range(max_size, 20, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - text_w // 2 - bbox[0], cy - text_h // 2 - bbox[1]), text, font=font, fill=fill)
            return


def create_share_image(photo, score, title):
    template_path = "assets/mall/mall.png"
    if not os.path.exists(template_path):
        return None
    template = Image.open(template_path).convert("RGBA")
    canvas = template.copy()
    photo = photo.convert("RGB")
    cropped = _cover_crop(photo, PHOTO_W, PHOTO_H)
    canvas.paste(cropped, (PHOTO_X, PHOTO_Y))
    draw = ImageDraw.Draw(canvas)
    score_str = f"{score:.0f}"
    score_size = 300 if len(score_str) == 3 else 360
    suffix_size = score_size // 3
    cy = SCORE_Y + SCORE_H // 2
    title_text = _strip_emoji(title).upper()
    _draw_class_text(draw, title_text, IMAGE_CENTER_X, CLASS_Y + CLASS_H // 2, PHOTO_W, SHARE_FILL)
    _draw_score_block(draw, score_str, IMAGE_CENTER_X, cy, _load_font(score_size), _load_font(suffix_size), SHARE_FILL)
    return canvas.convert("RGB")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    MAX_SIZE = 20 * 1024 * 1024  # 20 MB
    try:
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="Bilden är för stor (max 20 MB).")
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
    except HTTPException:
        raise
    except Exception as e:
        asyncio.create_task(logger.log_error("upload_error", str(e)))
        raise HTTPException(status_code=400, detail="Kunde inte läsa bilden.")

    try:
        result = predict(image)
    except Exception as e:
        asyncio.create_task(logger.log_error("predict_error", str(e)))
        raise HTTPException(status_code=500, detail="Analysen misslyckades, försök igen.")

    if result is None:
        asyncio.create_task(logger.log_error("no_face", "Inget ansikte hittades"))
        raise HTTPException(status_code=422, detail="Inget ansikte hittades.")

    if not result["mustache"]:
        return {
            "score": 0,
            "title": "🚫 Mustaschlös",
            "description": "Överläppen verkar för närvarande sakna tillräcklig auktoritet.",
            "mustache": False,
            "certificate": None,
        }

    p_epic        = result["p_epic"]
    p_medium      = result["p_medium"]
    p_medium_thin = result["p_medium_thin"]
    p_thin        = result["p_thin"]

    if p_epic >= 0.99:
        score = 100.0
    else:
        thin_score = 10 + (1 - p_thin) * 10
        score = p_epic * 95 + p_medium * 80 + p_medium_thin * 50 + p_thin * thin_score
    score = float(np.clip(score, 0, 100))

    title, description, _ = classify_epicness(score)

    share_image = create_share_image(image, score, title)
    certificate_b64 = None
    if share_image is not None:
        buf = io.BytesIO()
        share_image.save(buf, format="PNG")
        certificate_b64 = base64.b64encode(buf.getvalue()).decode()

    asyncio.create_task(logger.log_analysis(score))

    return {
        "score": round(score),
        "title": title,
        "description": description,
        "mustache": True,
        "certificate": certificate_b64,
    }


app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
