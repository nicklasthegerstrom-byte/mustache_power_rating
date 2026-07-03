const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultCard = document.getElementById('result-card');
const resultImage = document.getElementById('result-image');
const resultTitle = document.getElementById('result-title');
const resultScore = document.getElementById('result-score');

fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    analyzeBtn.hidden = false;
    resultCard.hidden = true;
});

analyzeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.textContent = '⏳ Analyserar...';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Något gick fel.');
            return;
        }

        const data = await res.json();

        resultTitle.textContent = data.title;
        resultScore.textContent = data.mustache
            ? `Mustaschkraft: ${data.score} / 100`
            : 'Ingen mustasch hittades.';

        resultCard.hidden = false;
        resultCard.scrollIntoView({ behavior: 'smooth' });

    } catch (e) {
        alert('Kunde inte nå servern.');
    } finally {
        analyzeBtn.textContent = '🔬 Analysera';
        analyzeBtn.disabled = false;
    }
});
