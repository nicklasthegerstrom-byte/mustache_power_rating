import os
import io
import glob
import time
import torch
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
from pillow_heif import register_heif_opener

register_heif_opener()  # gör att Image.open() också klarar .heic (iPhone-kamerafoton)

DEBUG_MODE = os.environ.get("MUSTASCH_DEBUG") != "0"

MODEL_CLASSES = 4  # byt till 3 för 3-klassmodellen

mtcnn = MTCNN(image_size=178, margin=40, post_process=False)

# Separat MTCNN-instans för ansiktsigenkänning (blocklista) — FaceNet förväntar
# sig 160x160, en annan storlek än mustasch-modellernas 178x178.
mtcnn_face = MTCNN(image_size=160, margin=0, post_process=True)
facenet = InceptionResnetV1(pretrained="vggface2").eval()

BLOCKLIST_THRESHOLD = 0.65
BLOCKLIST_ROOT = "assets/blocklist_reference"

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Mustaschkampens mustaschanalys",
    page_icon="🥸",
    layout="centered"
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():
    mustache_model = tf.keras.models.load_model(
        "models/mustache_detector_3.keras"
    )
    if MODEL_CLASSES == 4:
        epic_model = tf.keras.models.load_model(
            "models/epic_detector_4class_1.keras"
        )
    else:
        epic_model = tf.keras.models.load_model(
            "models/epic_detector_3class_test.keras"
        )
    return mustache_model, epic_model


mustache_model, epic_model = load_models()


@st.cache_resource
def load_blocklist_embeddings():
    """Bygger referensembeddings för blocklistan en gång vid appstart.

    Skannar ALLA undermappar under BLOCKLIST_ROOT (en mapp per spärrad
    person) — lägg bara till en ny mapp med referensbilder för att blockera
    fler personer, ingen kodändring behövs.
    """
    if not os.path.isdir(BLOCKLIST_ROOT):
        return []

    all_paths = []
    for person_dir in glob.glob(os.path.join(BLOCKLIST_ROOT, "*")):
        if not os.path.isdir(person_dir):
            continue
        all_paths += glob.glob(os.path.join(person_dir, "*.jpg"))
        all_paths += glob.glob(os.path.join(person_dir, "*.jpeg"))
        all_paths += glob.glob(os.path.join(person_dir, "*.png"))

    embeddings = []
    for path in all_paths:
        try:
            face = mtcnn_face(Image.open(path).convert("RGB"))
        except Exception:
            continue
        if face is None:
            continue
        with torch.no_grad():
            emb = facenet(face.unsqueeze(0)).squeeze(0).numpy()
        embeddings.append(emb)

    return embeddings


# Ordningen image_dataset_from_directory sorterar mappar alfabetiskt.
CLASS_NAMES = ["epic", "medium", "medium_thin", "thin"] if MODEL_CLASSES == 4 else ["epic", "medium", "thin"]

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = (178, 178)

# --------------------------------------------------
# Header
# --------------------------------------------------

_, logo_col, _ = st.columns([1, 2, 1])
with logo_col:
    st.image("assets/logo.png", width=600)


st.warning("🧪 TESTVERSION — 3-klassmodell (episk/respektabel/tunn)", icon="🧪")

# Laddas (och cachas) först HÄR, efter headern, så headern hinner synas
# innan kallstartspausen — istället för en blank sida.
with st.spinner("🥸 Mustaschinstrument laddas..."):
    blocklist_embeddings = load_blocklist_embeddings()

# --------------------------------------------------
# Upload
# --------------------------------------------------

main_card = st.container(border=True, key="main_card")
with main_card:
    upl_col, info_col = st.columns([10, 1])
    with upl_col:
        uploaded_file = st.file_uploader(
            "Skicka in provet för analys",
            type=["jpg", "jpeg", "png", "heic", "heif"],
            key="file_uploader"
        )
    with info_col:
        st.write("")
        with st.popover("ℹ️"):
            st.markdown(
                "<p style='font-size:0.75rem; margin:0;'>"
                "<strong>Tips för bästa resultat</strong><br>"
                "Ta bilden rakt framifrån<br>"
                "Håll munnen stängd<br>"
                "Se till att ha bra belysning<br>"
                "Se till att mustaschen syns tydligt<br>"
                "Undvik motion blur eller oskärpa"
                "</p>",
                unsafe_allow_html=True
            )

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def weighted_epic_score(p_epic, p_medium, p_thin):
    """Viktad poäng 0-100 baserat på de tre klassernas sannolikheter.

    Epic och medium behandlas som EN kontinuerlig skala (ingen "vem leder"-
    gren mellan dem) — annars uppstår en kliff exakt där epic/medium byter
    plats som störst, vilket gav motsägelsefulla resultat (mer epic kunde ge
    LÄGRE poäng än mindre epic, bara pga vilken som råkade vara en hårsbredd
    större). Bara thin är en genuint annan kategori ("golvet") värd en
    separat gren.
    """
    p_max = max(p_epic, p_medium, p_thin)

    if p_max < 0.40:
        if p_thin == p_max:
            return 20.0
        return 40.0

    if p_thin == p_max:
        score = p_thin * 7 + p_epic * 100 + p_medium * 50
        if p_max < 0.7:
            score *= 0.92  # generöst — appen ska vara kul att dela, inte sträng
    else:
        # Epic och medium är samma kontinuerliga skala — ingen kliff mellan dem.
        # Thin straffar mot ett glidande "tak" (mellan 65 och 100) baserat på
        # hur mycket av epic/medium-blandningen som faktiskt finns, istället
        # för att alltid anta att motparten var en fullvärdig epic-claim.
        epic_medium_sum = p_epic + p_medium
        effective_anchor = (
            (p_epic * 100 + p_medium * 65) / epic_medium_sum
            if epic_medium_sum > 0 else 100
        )
        score = (p_epic ** 1.1) * 100 + p_medium * 65 - p_thin * effective_anchor * 0.5

    score = float(np.clip(score, 0, 100))

    # Global "var snäll"-kurva — lyfter mellanzonen (störst lyft runt 30-40),
    # rör knappt redan-säkra extremfall (0 och 100 påverkas inte).
    score = 100 * (score / 100) ** 0.85

    return float(np.clip(score, 0, 100))


def weighted_epic_score_4class(p_epic, p_medium, p_medium_thin, p_thin):
    if p_epic >= 0.99:
        return 100.0
    score = p_epic * 95 + p_medium * 80 + p_medium_thin * 50 + p_thin * 15
    return float(np.clip(score, 0, 100))


def compress_top_end(score, floor=95.0, new_floor=85.0, gamma=2.5):
    """Mappar [floor, 100] till [new_floor, 100] med en potenskurva.

    Bara score==100 (exakt 1.0/0.0/0.0, händer i praktiken vid float32-
    underflow på extremt övertygande bilder) förblir 100 — allt annat i
    intervallet pressas tydligt nedåt, så mindre klumpar sig i toppen.
    """
    if score < floor:
        return score
    frac = (score - floor) / (100.0 - floor)
    curved = frac ** gamma
    return new_floor + (100.0 - new_floor) * curved


def is_blocked(image, reference_embeddings, threshold=BLOCKLIST_THRESHOLD):
    """Jämför ett uppladdat ansikte mot blocklistans referensembeddings."""
    face = mtcnn_face(image.convert("RGB"))
    if face is None or not reference_embeddings:
        return False, 0.0

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0)).squeeze(0).numpy()

    similarities = [
        float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        for ref in reference_embeddings
    ]
    best = max(similarities)
    return best >= threshold, best


def prepare_image(image):
    """Returnerar None om inget ansikte hittas — ingen gissning på textur/bakgrund."""
    image = image.convert("RGB")

    face = mtcnn(image)

    if face is None:
        return None

    # post_process=False → tensor i [0, 255] direkt
    arr = face.permute(1, 2, 0).numpy().astype(np.uint8)
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_epicness(score):
    if score >= 95:
        return (
            "🏆 En legendarisk mustasch",
            "Våra experter är mållösa. Mustaschen föreslås statligt kulturarvsskydd.",
            "assets/abbe/video/legendarisk.mp4"
        )

    elif score >= 80:
        return (
            "🔥 En episk mustasch",
            "En mustasch med styrka. Kraft. Själ.",
            "assets/abbe/video/episk.mp4"
        )

    elif score >= 60:
        return (
            "🎩 En respektabel mustasch",
            "En värdig representant för svensk mustaschkultur.",
            "assets/abbe/video/respektabel.mp4"
        )

    elif score >= 25:
        return (
            "🌱 En lovande mustasch",
            "Ett litet steg för överläppen. Ett stort steg för mänskligheten.",
            "assets/abbe/video/lovande.mp4"
        )

    else:
        return (
            "🪶 En fjunig mustasch",
            "Mustaschen existerar mest som ett teoretiskt koncept.",
            "assets/abbe/video/fjunig.mp4"
        )


def _load_font(size):
    candidates = [
        "assets/fonts/FalstaffMTStd.otf",
        "assets/fonts/Akzidenz-grotesk-black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _load_title_font(size):
    candidates = [
        "assets/fonts/Gotham Medium.otf",
        "assets/fonts/Akzidenz-grotesk-black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _cover_crop(img, target_w, target_h):
    """Beskär bilden centrerat så den fyller målstorleken utan att förvrängas."""
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    return img.resize((target_w, target_h))


def _draw_centered_text(draw, text, box, font, fill):
    x, y, w, h = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos_x = x + (w - text_w) // 2 - bbox[0]
    pos_y = y + (h - text_h) // 2 - bbox[1]
    draw.text((pos_x, pos_y), text, font=font, fill=fill)


def _draw_right_aligned_text(draw, text, box, font, fill):
    x, y, w, h = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos_x = x + w - text_w - bbox[0]
    pos_y = y + (h - text_h) // 2 - bbox[1]
    draw.text((pos_x, pos_y), text, font=font, fill=fill)


def _strip_emoji(text):
    return "".join(ch for ch in text if ch.isascii() or ch.isalpha()).strip()


PHOTO_X, PHOTO_Y, PHOTO_W, PHOTO_H = 428, 548, 1001, 1019
SCORE_Y, SCORE_H = 1745, 276
CLASS_X, CLASS_Y, CLASS_W, CLASS_H = 704, 1620, 462, 45
IMAGE_CENTER_X = 920
SHARE_FILL = (54, 88, 159)


def _draw_score_block(draw, score_str, cx, cy, score_font, suffix_font, fill):
    """Rita score + /100 som ett horisontellt centrerat block."""
    gap = 20
    sb = draw.textbbox((0, 0), score_str, font=score_font)
    score_w, score_h = sb[2]-sb[0], sb[3]-sb[1]
    fb = draw.textbbox((0, 0), "/100", font=suffix_font)
    suffix_w, suffix_h = fb[2]-fb[0], fb[3]-fb[1]
    total_w = score_w + gap + suffix_w
    start_x = cx - total_w // 2
    draw.text((start_x - sb[0], cy - score_h//2 - sb[1]), score_str, font=score_font, fill=fill)
    suffix_x = start_x + score_w + gap
    suffix_y = cy + score_h//2 - suffix_h - sb[1]
    draw.text((suffix_x - fb[0], suffix_y - fb[1]), "/100", font=suffix_font, fill=fill)


def _draw_class_text(draw, text, cx, cy, max_w, fill, max_size=76):
    """Rita klasstexter centrerad på (cx, cy), krymper om texten är för lång."""
    font_path = "assets/fonts/Gotham Bold.otf"
    for size in range(max_size, 20, -2):
        try: font = ImageFont.truetype(font_path, size)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.text((cx - text_w//2 - bbox[0], cy - text_h//2 - bbox[1]), text, font=font, fill=fill)
            return


def create_share_image(photo, score, title, description="", template_path="assets/mall/mall.png"):
    """Klistrar in foto + klass + poäng i den färdigdesignade mallen.

    Returnerar None om mallen saknas, så anroparen kan visa ett snyggt
    felmeddelande istället för att krascha hela appen.
    """
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
    _draw_class_text(draw, title_text, IMAGE_CENTER_X, CLASS_Y + CLASS_H // 2, PHOTO_W, SHARE_FILL, max_size=76)
    _draw_score_block(draw, score_str, IMAGE_CENTER_X, cy, _load_font(score_size), _load_font(suffix_size), SHARE_FILL)

    return canvas.convert("RGB")


def animate_scanner():
    labels = [
        "🔬 Skannar överläppsregionen...",
        "🧬 Analyserar hårdensitetsmatris...",
        "⚗️  Korsreferens mot mustaschdatabasen...",
        "📊 Beräknar mustaschkraft...",
    ]

    progress_bar = st.progress(0)
    status = st.empty()
    meter1 = st.empty()
    meter2 = st.empty()

    for i, label in enumerate(labels):
        status.markdown(f"**{label}**")

        for v in list(range(0, 101, 5)) + list(range(100, -1, -5)):
            meter1.progress(v / 100)
            meter2.progress(abs(v - 100) / 100)
            time.sleep(0.015)

        progress_bar.progress((i + 1) / len(labels))

    status.empty()
    meter1.empty()
    meter2.empty()
    progress_bar.empty()

# --------------------------------------------------
# Analysis
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)  # rättar telefonfoton som annars blir sidvända
    except Exception as e:
        print(f"[LOGG] Kunde inte öppna uppladdad fil: {e}", flush=True)
        st.error("❌ KUNDE INTE LÄSA FILEN — prova en annan bild (JPG/PNG).")
        st.stop()

    print(f"[LOGG] Bild mottagen: {uploaded_file.name}, {uploaded_file.size} bytes, storlek {image.size}", flush=True)

    with main_card:
        left_col, right_col = st.columns([1, 1])
        left_slot = left_col.empty()
        right_slot = right_col.empty()

        with left_slot.container():
            st.image(image, caption="Inskickat prov", width="stretch")

        with right_slot.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze = st.button("Starta analys", width="stretch", key="analyze_button")

        extra_area = st.empty()

        if analyze:
            print("[LOGG] Analys startad.", flush=True)

            blocked, block_similarity = is_blocked(image, blocklist_embeddings)
            print(f"[LOGG] Blocklist-kontroll klar: blocked={blocked}, similarity={block_similarity:.3f}", flush=True)

            if blocked:
                with right_slot.container():
                    st.error("❌ DEN HÄR PERSONEN KAN INTE ANALYSERAS")
                    st.write(
                        "Utlåtande: Specimen matchar en spärrad profil och "
                        "nekas certifiering."
                    )
                st.stop()

            img_array = prepare_image(image)
            print(f"[LOGG] Ansiktsdetektion (prepare_image) klar: {'ansikte hittat' if img_array is not None else 'INGET ansikte'}", flush=True)

            if img_array is None:
                with right_slot.container():
                    st.error("❌ INGET ANSIKTE UPPTÄCKT")
                    st.write(
                        "Utlåtande: Specimen innehåller inget identifierbart "
                        "mänskligt ansikte och kan inte certifieras."
                    )
                st.stop()

            with right_slot.container():
                animate_scanner()

            print("[LOGG] Kör mustache_model.predict...", flush=True)
            mustache_prob = float(
                mustache_model.predict(img_array, verbose=0)[0][0]
            )
            print(f"[LOGG] mustache_model klar: mustache_prob={mustache_prob:.4f}", flush=True)

            if mustache_prob < 0.4:
                p_epic, p_medium, p_medium_thin, p_thin = 0.0, 0.0, 0.0, 0.0
                epic_score = 0.0
                title, description, video_file = "🚫 Mustaschlös", (
                    "Överläppen verkar för närvarande sakna "
                    "tillräcklig auktoritet."
                ), "assets/abbe/video/ingen.mp4"
            else:
                print("[LOGG] Kör epic_model.predict...", flush=True)
                preds = epic_model.predict(img_array, verbose=0)[0]

                if MODEL_CLASSES == 4:
                    p_epic, p_medium, p_medium_thin, p_thin = float(preds[0]), float(preds[1]), float(preds[2]), float(preds[3])
                    print(f"[LOGG] epic_model klar: p_epic={p_epic:.4f}, p_medium={p_medium:.4f}, p_medium_thin={p_medium_thin:.4f}, p_thin={p_thin:.4f}", flush=True)
                    epic_score = weighted_epic_score_4class(p_epic, p_medium, p_medium_thin, p_thin)
                else:
                    p_epic, p_medium, p_medium_thin, p_thin = float(preds[0]), float(preds[1]), 0.0, float(preds[2])
                    print(f"[LOGG] epic_model klar: p_epic={p_epic:.4f}, p_medium={p_medium:.4f}, p_thin={p_thin:.4f}", flush=True)
                    epic_for_score, medium_for_score, thin_for_score = p_epic, p_medium, p_thin
                    if p_epic >= 0.99:
                        epic_for_score, medium_for_score, thin_for_score = 1.0, 0.0, 0.0
                    epic_score = weighted_epic_score(epic_for_score, medium_for_score, thin_for_score)
                    epic_score = compress_top_end(epic_score)

                print(f"[LOGG] Poäng beräknad: {epic_score:.2f}", flush=True)
                title, description, video_file = classify_epicness(epic_score)

            print("[LOGG] Genererar delningsbild...", flush=True)
            share_image = create_share_image(image, epic_score, title, description)
            print(f"[LOGG] Delningsbild klar: {'OK' if share_image is not None else 'MISSLYCKADES'}", flush=True)

            # Göm sidokolumnerna, visa allt i extra_area (full bredd)
            left_slot.empty()
            right_slot.empty()

            with extra_area.container():
                FALSTAFF = "Falstaff, serif"
                GOTHAM = "'Gotham Medium', sans-serif"
                if video_file and os.path.exists(video_file):
                    st.video(video_file)
                st.markdown(
                    f"<p style='font-family: {FALSTAFF}; font-size: 2rem; color: #365899; margin-bottom: 0; text-align: center;'>{title}</p>"
                    f'<p style="font-family: Gotham Medium, sans-serif; font-size: 1.6rem; color: #444; margin: 4px 0 6px 0; text-align: center;">Mustaschstyrka: {epic_score:.0f} / 100</p>'
                    f"<div class='mustasch-desc' style='font-family: {GOTHAM};'>{description}</div>",
                    unsafe_allow_html=True
                )
                if share_image is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    _, cert_col, _ = st.columns([1, 3, 1])
                    with cert_col:
                        st.image(share_image, use_container_width=True)
                if DEBUG_MODE:
                    st.write("mustasch_sannolikhet:", mustache_prob)
                    st.write("p_episk:", p_epic)
                    st.write("p_respektabel:", p_medium)
                    if MODEL_CLASSES == 4:
                        st.write("p_lovande:", p_medium_thin)
                    st.write("p_tunn:", p_thin)
                    st.write("mustaschkraft_poäng:", epic_score)

                st.divider()
                st.markdown(
                    "<p style='text-align: center; color: gray; font-size: 0.875rem;'>"
                    "Resultat certifierat av "
                    "Mustaschkampens legitimerade mustaschexperter™"
                    "</p>",
                    unsafe_allow_html=True
                )

                print("[LOGG] Analys klar, resultat visat.", flush=True)

st.markdown(
    "<p style='font-family: Montserrat, sans-serif; text-align: center; color: #555; font-size: 0.9rem; margin-top: 2rem;'>"
    "Swisha ditt bidrag till <strong style='color: #1a1a1a;'>900 10 17</strong>.<br>"
    "Alla gåvor är välkomna, varje krona gör skillnad i kampen mot Sveriges vanligaste cancersjukdom."
    "</p>",
    unsafe_allow_html=True
)
