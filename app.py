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
    epic_model = tf.keras.models.load_model(
        "models/epic_detector.keras"
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


blocklist_embeddings = load_blocklist_embeddings()

# Ordningen image_dataset_from_directory sorterar mappar alfabetiskt:
# epic, medium, thin
CLASS_NAMES = ["epic", "medium", "thin"]

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = (178, 178)

# --------------------------------------------------
# Header
# --------------------------------------------------

_, logo_col, _ = st.columns([1, 2, 1])
with logo_col:
    st.image("assets/abbe/logo.png", width=600)

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.875rem;'>"
    "Officiell ansiktshårsbedömning "
    "av Mustaschkampens legitimerade mustaschexperter"
    "</p>",
    unsafe_allow_html=True
)

st.warning("🧪 TESTVERSION — 3-klassmodell (episk/respektabel/tunn)", icon="🧪")

# --------------------------------------------------
# Upload
# --------------------------------------------------

main_card = st.container(border=True, key="main_card")
with main_card:
    uploaded_file = st.file_uploader(
        "Skicka in provet för analys",
        type=["jpg", "jpeg", "png", "heic", "heif"]
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
        # Praktiskt taget jämnt delat (nära 1/3 var) — modellen har ingen
        # riktig åsikt. Straffa inte total osäkerhet som om det var ett
        # dåligt resultat.
        return 40.0

    if p_thin == p_max:
        score = p_thin * 8 + p_epic * 100 + p_medium * 50
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
        score = p_epic * 100 + p_medium * 65 - p_thin * effective_anchor
        if p_thin > 0.05:
            score *= 0.92

    score = float(np.clip(score, 0, 100))

    # Global "var snäll"-kurva — lyfter mellanzonen (störst lyft runt 30-40),
    # rör knappt redan-säkra extremfall (0 och 100 påverkas inte).
    score = 100 * (score / 100) ** 0.85

    return float(np.clip(score, 0, 100))


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
            "🏆 Legendarisk",
            "Mustaschkampens experter är mållösa.",
            "assets/abbe/legendarisk.mp4"
        )

    elif score >= 80:
        return (
            "🔥 Episk",
            "En mustasch av ovanligt hög kaliber.",
            "assets/abbe/episk.mp4"
        )

    elif score >= 60:
        return (
            "🎩 Respektabel",
            "Godkänd. Inte historisk, men godkänd.",
            "assets/svenska/respektabel.mp4"
        )

    elif score >= 30:
        return (
            "🌱 Lovande",
            "Mustaschtillväxten befinner sig fortfarande i betatest.",
            "assets/abbe/lovande.mp4"
        )

    else:
        return (
            "🪶 Fjunig",
            "Mustaschen existerar mest som ett teoretiskt koncept.",
            "assets/fjunig.mp4"
        )


def _load_font(size):
    candidates = [
        "assets/fonts/OpenSans-Bold.ttf",  # buntad med appen — funkar oavsett plattform
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


# Koordinater från mallen (assets/mall.png)
PHOTO_X, PHOTO_Y, PHOTO_W, PHOTO_H = 150, 275, 810, 610
SCORE_X, SCORE_Y, SCORE_W, SCORE_H = 150, 960, 810, 120
TITLE_X, TITLE_Y, TITLE_W, TITLE_H = 110, 1125, 900, 130


def create_share_image(photo, score, title, description="", template_path="assets/mall.png"):
    """Klistrar in foto + poäng + titel + underrubrik i den färdigdesignade mallen.

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

    score_font = _load_font(100)
    title_font = _load_font(44)
    description_font = _load_font(26)

    # Skriv hela "XX / 100" centrerat i rutan (täcker mallens inbrända "____/100").
    _draw_centered_text(
        draw, f"{score:.0f} / 100",
        (SCORE_X, SCORE_Y, SCORE_W, SCORE_H),
        score_font, "#d4a843"
    )

    # Titel och underrubrik staplade i samma ruta — titel överst, beskrivning under.
    title_text = _strip_emoji(title)
    desc_text = _strip_emoji(description)

    _draw_centered_text(
        draw, title_text,
        (TITLE_X, TITLE_Y, TITLE_W, TITLE_H // 2 + 15),
        title_font, "white"
    )
    _draw_centered_text(
        draw, desc_text,
        (TITLE_X, TITLE_Y + TITLE_H // 2 + 5, TITLE_W, TITLE_H // 2 - 5),
        description_font, "#9fb3c8"
    )

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
    except Exception:
        st.error("❌ KUNDE INTE LÄSA FILEN — prova en annan bild (JPG/PNG).")
        st.stop()

    with main_card:
        left_col, right_col = st.columns([1, 1])
        left_slot = left_col.empty()
        right_slot = right_col.empty()

        with left_slot.container():
            st.image(image, caption="Inskickat prov", use_container_width=True)

        with right_slot.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze = st.button("🔬 Starta analys", use_container_width=True)

        extra_area = st.empty()  # video/debug-info/footer, fylls efter analys

        if analyze:

            blocked, block_similarity = is_blocked(image, blocklist_embeddings)

            if blocked:
                with right_slot.container():
                    st.error("❌ DEN HÄR PERSONEN KAN INTE ANALYSERAS")
                    st.write(
                        "Utlåtande: Specimen matchar en spärrad profil och "
                        "nekas certifiering."
                    )
                st.stop()

            img_array = prepare_image(image)

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

            mustache_prob = float(
                mustache_model.predict(img_array, verbose=0)[0][0]
            )

            if mustache_prob < 0.4:
                with right_slot.container():
                    st.error("❌ INGEN CERTIFIERAD MUSTASCH UPPTÄCKT")
                    st.write(
                        "Utlåtande: Överläppen verkar för närvarande "
                        "sakna tillräcklig auktoritet."
                    )

                with extra_area.container():
                    st.write("mustasch_sannolikhet:", mustache_prob)
                    if os.path.exists("assets/abbe/ingen.mp4"):
                        st.video("assets/abbe/ingen.mp4")

            else:
                preds = epic_model.predict(img_array, verbose=0)[0]
                p_epic, p_medium, p_thin = float(preds[0]), float(preds[1]), float(preds[2])

                epic_score = weighted_epic_score(p_epic, p_medium, p_thin)

                title, description, video_file = classify_epicness(epic_score)

                share_image = create_share_image(image, epic_score, title, description)

                # VÄNSTER: bytut foto mot resultatbilden
                with left_slot.container():
                    if share_image is not None:
                        st.image(share_image, use_container_width=True)
                    else:
                        st.warning("⚠️ Kunde inte generera delningsbild (mallen saknas).")
                        st.image(image, caption="Inskickat prov", use_container_width=True)

                # HÖGER: byt ut knapp/mätare mot titel + nedladdning
                with right_slot.container():
                    st.subheader(title)
                    st.write(description)

                    if share_image is not None:
                        buf = io.BytesIO()
                        share_image.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            "⬇️ Ladda ner bild",
                            data=buf,
                            file_name="mustaschkraft.png",
                            mime="image/png",
                            use_container_width=True
                        )

                    if epic_score >= 95:
                        st.balloons()

                with extra_area.container():
                    st.write("mustasch_sannolikhet:", mustache_prob)
                    st.write("p_episk:", p_epic)
                    st.write("p_respektabel:", p_medium)
                    st.write("p_tunn:", p_thin)
                    st.write("mustaschkraft_poäng:", epic_score)

                    if video_file and os.path.exists(video_file):
                        st.video(video_file)

                    st.divider()
                    st.markdown(
                        "<p style='text-align: center; color: gray; font-size: 0.875rem;'>"
                        "Resultat certifierat av "
                        "Mustaschkampens legitimerade mustaschexperter™"
                        "</p>",
                        unsafe_allow_html=True
                    )
