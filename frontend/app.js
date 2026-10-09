const API_BASE = window.location.origin;
const STUDENT_ID = "S001-ALPHA"; // Demo Student

// App State
let appState = {
    sourceId: null,
    quiz: [],
    index: 0,
    difficulty: 'medium',
    isLocked: false,
    score: 0,
    attempts: 0,
    correctCount: 0,
    history: [] // {correct: bool, level: string}
};

// DOM Cache
const dom = {
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('pdf-upload'),
    uploadStatus: document.getElementById('upload-status'),
    genSection: document.getElementById('gen-section'),
    genStatus: document.getElementById('gen-status'),
    btnGenerate: document.getElementById('btn-generate'),
    btnReset: document.getElementById('btn-reset'),
    
    emptyState: document.getElementById('empty-state'),
    quizRunner: document.getElementById('quiz-runner'),
    resultsScreen: document.getElementById('results-screen'),
    
    qText: document.getElementById('q-text'),
    optionsStack: document.getElementById('options-stack'),
    diffBadge: document.getElementById('diff-badge'),
    currIdx: document.getElementById('curr-idx'),
    totalIdx: document.getElementById('total-idx'),
    progressBar: document.getElementById('progress-bar'),
    
    feedback: document.getElementById('feedback-area'),
    btnNext: document.getElementById('btn-next'),
    btnSubmit: document.getElementById('btn-submit'),
    overlay: document.getElementById('overlay-container'),

    // Adaptive Panel
    progressPanel: document.getElementById('student-progress-panel'),
    statAttempted: document.getElementById('stat-attempted'),
    statAccuracy: document.getElementById('stat-accuracy'),
    statLevel: document.getElementById('stat-level'),
    chartContainer: document.getElementById('performance-chart-container')
};

// State for selection
let selectedValue = null;
let selectedCardElement = null;

// --- Ingestion Logic ---
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
dom.uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); dom.uploadZone.classList.add('dragover'); });
dom.uploadZone.addEventListener('dragleave', () => dom.uploadZone.classList.remove('dragover'));
dom.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});

dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
});

// Helper for overlay
function showProcessing(text) {
    document.getElementById('processing-overlay').classList.remove('hidden');
    document.getElementById('overlay-text').textContent = text;
}
function hideProcessing() {
    document.getElementById('processing-overlay').classList.add('hidden');
}

async function handleUpload(file) {
    if (file.type !== 'application/pdf') return notifyUpload("Error: PDF required", "error");
    
    showProcessing("Analyzing Knowledge Layers...");
    notifyUpload("Parsing knowledge...", "pending");
    
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error();
        const data = await res.json();
        
        appState.sourceId = data.source_id;
        notifyUpload(`Ready: ${data.chunks_extracted} layers found.`, "success");
        
        // Show status panel
        document.getElementById('doc-status-panel').classList.remove('hidden');
        document.getElementById('status-filename').textContent = data.filename;
        document.getElementById('status-subject').textContent = data.subject || "General";
        document.getElementById('status-grade').textContent = data.grade || "AI Derived";
        document.getElementById('status-pages').textContent = data.pages_count;
        document.getElementById('status-chunks').textContent = data.chunks_extracted;
        document.getElementById('status-questions').textContent = "-";

        dom.genSection.classList.remove('disabled');
        dom.genSection.classList.add('fade-in');
    } catch (e) {
        notifyUpload("Server offline.", "error");
    } finally {
        hideProcessing();
    }
}

function notifyUpload(msg, type) {
    dom.uploadStatus.innerHTML = `<div class="status-pill ${type}">${msg}</div>`;
}

// --- Generation Logic ---
dom.btnGenerate.addEventListener('click', async () => {
    if (appState.isLocked) return;
    
    appState.isLocked = true;
    dom.btnGenerate.disabled = true;
    showProcessing("Igniting AI Assessment Engine...");
    if (window.lucide) lucide.createIcons();
    
    try {
        if (appState.sourceId) {
            await fetch(`${API_BASE}/generate-quiz`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_id: appState.sourceId })
            }).catch(() => {});
        }
        
        // Read filters
        const topic = document.getElementById('filter-topic')?.value || '';
        const diff = document.getElementById('filter-difficulty')?.value || '';
        let url = `${API_BASE}/quiz?`;
        if (topic) url += `topic=${encodeURIComponent(topic)}&`;
        if (diff) url += `difficulty=${encodeURIComponent(diff)}`;

        // Fetch questions
        const qRes = await fetch(url);
        let questions = await qRes.json();

        // Fallback if filter returned empty
        if (!Array.isArray(questions) || questions.length === 0) {
            const fallbackRes = await fetch(`${API_BASE}/quiz`);
            questions = await fallbackRes.json();
        }

        // Default seed fallback if database is empty
        if (!Array.isArray(questions) || questions.length === 0) {
            questions = [
                {
                    id: "SEED-Q-001",
                    question: "Which pigment absorbs light energy in plant cells during photosynthesis?",
                    type: "MCQ",
                    options: ["Chlorophyll", "Carotenoid", "Anthocyanin", "Hemoglobin"],
                    answer: "Chlorophyll",
                    difficulty: "easy",
                    source_chunk_text: "Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy stored in glucose."
                },
                {
                    id: "SEED-Q-002",
                    question: "What is the primary energy transformation in photosynthesis?",
                    type: "MCQ",
                    options: ["Light energy into chemical energy", "Chemical energy into heat", "Nuclear energy into light", "Kinetic energy into electricity"],
                    answer: "Light energy into chemical energy",
                    difficulty: "medium",
                    source_chunk_text: "Photosynthesis converts solar light energy into chemical energy stored in molecular bonds."
                },
                {
                    id: "SEED-Q-003",
                    question: "In Reinforcement Learning, what does the Q-value Q(s, a) represent?",
                    type: "MCQ",
                    options: ["Expected cumulative reward for taking action a in state s", "Instant prediction error", "Number of neural network layers", "Learning rate multiplier"],
                    answer: "Expected cumulative reward for taking action a in state s",
                    difficulty: "hard",
                    source_chunk_text: "Deep Q-Learning (DQN) combines neural networks with Q-learning to approximate optimal Q-values."
                },
                {
                    id: "SEED-Q-004",
                    question: "Which framework estimates student skill ability theta based on response accuracy?",
                    type: "MCQ",
                    options: ["Item Response Theory (IRT)", "Linear Regression", "K-Means Clustering", "Fourier Transform"],
                    answer: "Item Response Theory (IRT)",
                    difficulty: "medium",
                    source_chunk_text: "Item Response Theory (IRT) models student skill ability theta and question difficulty b."
                }
            ];
        }
        
        appState.quiz = questions;
        const qCountElem = document.getElementById('status-questions');
        if (qCountElem) qCountElem.textContent = appState.quiz.length;

        initQuiz();
    } catch (e) {
        console.error("Quiz Ignite Error:", e);
        dom.genStatus.innerHTML = `<div class="status-pill error">Notice: Loaded local fallback questions.</div>`;
        appState.quiz = [
            {
                id: "SEED-Q-001",
                question: "Which pigment absorbs light energy in plant cells during photosynthesis?",
                type: "MCQ",
                options: ["Chlorophyll", "Carotenoid", "Anthocyanin", "Hemoglobin"],
                answer: "Chlorophyll",
                difficulty: "easy",
                source_chunk_text: "Photosynthesis uses chlorophyll to absorb light energy."
            }
        ];
        initQuiz();
    } finally {
        dom.btnGenerate.disabled = false;
        appState.isLocked = false;
        hideProcessing();
        if (window.lucide) lucide.createIcons();
    }
});


// ... same quiz logic ...

function finish() {
    dom.progressBar.style.width = '100%';
    dom.quizRunner.classList.add('hidden');
    dom.resultsScreen.classList.remove('hidden');
    dom.progressPanel.classList.add('hidden'); // Hide partial tracker to focus on final
    
    // Populate Final Stats
    document.getElementById('final-score').textContent = `${appState.correctCount} / ${appState.attempts}`;
    const acc = appState.attempts > 0 ? Math.round((appState.correctCount / appState.attempts) * 100) : 0;
    document.getElementById('final-accuracy').textContent = `${acc}%`;
    const lastLevel = appState.history.length > 0 ? appState.history[appState.history.length-1].level : 'medium';
    document.getElementById('final-level').textContent = lastLevel.charAt(0).toUpperCase() + lastLevel.slice(1);
    
    confetti({ particleCount: 150, spread: 100, origin: { y: 0.5 } });
    lucide.createIcons();
}

// --- Quiz Logic ---
function initQuiz() {
    dom.emptyState.classList.add('hidden');
    dom.genSection.classList.add('hidden');
    dom.quizRunner.classList.remove('hidden');
    dom.progressPanel.classList.remove('hidden');
    
    // Reset Stats
    appState.index = 0;
    appState.score = 0;
    appState.attempts = 0;
    appState.correctCount = 0;
    appState.history = [];
    
    updateProgressUI();
    dom.chartContainer.innerHTML = "";
    dom.totalIdx.textContent = appState.quiz.length;
    renderQuestion();
}

function renderQuestion() {
    appState.isLocked = false;
    selectedValue = null;
    selectedCardElement = null;
    
    dom.btnNext.classList.add('hidden');
    dom.btnSubmit.classList.remove('hidden');
    dom.btnSubmit.disabled = true;
    dom.feedback.innerHTML = "";
    document.getElementById('source-trace-panel').classList.add('hidden');
    
    const q = appState.quiz[appState.index];
    dom.qText.textContent = q.question;
    dom.currIdx.textContent = appState.index + 1;
    
    // Progress bar
    const progress = ((appState.index) / appState.quiz.length) * 100;
    dom.progressBar.style.width = `${progress}%`;
    
    // UI Theme for difficulty
    dom.diffBadge.className = `badge ${q.difficulty}`;
    dom.diffBadge.textContent = q.difficulty;
    
    dom.optionsStack.innerHTML = "";
    q.options.forEach((opt, idx) => {
        const card = document.createElement('div');
        card.className = "option-card fade-in";
        card.style.animationDelay = `${idx * 0.1}s`;
        card.innerHTML = `
            <div class="option-index">${String.fromCharCode(65 + idx)}</div>
            <div class="option-text">${opt}</div>
        `;
        card.onclick = () => {
            if (appState.isLocked) return;
            if (selectedCardElement) selectedCardElement.classList.remove('selected');
            selectedValue = opt;
            selectedCardElement = card;
            card.classList.add('selected');
            dom.btnSubmit.disabled = false;
        };
        dom.optionsStack.appendChild(card);
    });
}

dom.btnSubmit.addEventListener('click', async () => {
    if (!selectedValue || appState.isLocked) return;
    const q = appState.quiz[appState.index];
    await submit(selectedValue, selectedCardElement, q);
});

async function submit(val, el, q) {
    appState.isLocked = true;
    dom.btnSubmit.classList.add('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/submit-answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                student_id: STUDENT_ID,
                question_id: q.id,
                selected_answer: val
            })
        });
        const data = await res.json();
        
        // Update Stats
        appState.attempts++;
        if (data.is_correct) appState.correctCount++;
        appState.history.push({ correct: data.is_correct, level: data.next_recommended_difficulty });
        updateProgressUI();
        addChartBar(data.is_correct, data.next_recommended_difficulty);

        // Feedback details with RL tag
        const diffText = data.original_difficulty === data.next_recommended_difficulty 
            ? `Difficulty: ${data.original_difficulty} (RL Policy Stable)`
            : `Difficulty: ${data.original_difficulty} → ${data.next_recommended_difficulty} (RL Policy Dynamic Shift)`;
        
        const sourceText = `Source: ${data.source_chunk_id.split('_').slice(-2).join(' ')}`;

        if (data.is_correct) {
            el.classList.add('correct');
            dom.feedback.innerHTML = `
                <div class="feedback-msg correct" style="flex-direction: column; align-items: flex-start;">
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <i data-lucide="check-circle-2"></i> ✅ Correct!
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.5rem; font-weight: 500;">
                        ${diffText} | ${sourceText}
                    </div>
                </div>`;
            confetti({ particleCount: 30, spread: 60, origin: { y: 0.8 } });
        } else {
            el.classList.add('wrong');
            dom.feedback.innerHTML = `
                <div class="feedback-msg wrong" style="flex-direction: column; align-items: flex-start;">
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <i data-lucide="x-circle"></i> ❌ Incorrect
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.5rem; font-weight: 500;">
                        Correct Answer: <strong>${data.correct_answer}</strong> <br>
                        ${diffText} | ${sourceText}
                    </div>
                </div>`;
        }
        
        lucide.createIcons();
        
        // Show Source Traceability
        document.getElementById('source-trace-panel').classList.remove('hidden');
        document.getElementById('source-text-display').textContent = `"${q.source_chunk_text}"`;
        document.getElementById('source-id-display').textContent = `Chunk ID: ${q.source_chunk_id}`;
        lucide.createIcons();

        dom.btnNext.classList.remove('hidden');
        
    } catch (e) {
        console.error(e);
        appState.isLocked = false;
        dom.btnSubmit.classList.remove('hidden');
    }
}

let bktRadarChart = null;

async function updateProgressUI() {
    dom.statAttempted.textContent = appState.attempts;
    const acc = appState.attempts > 0 ? Math.round((appState.correctCount / appState.attempts) * 100) : 0;
    dom.statAccuracy.textContent = `${acc}%`;
    
    // Get current difficulty from last entry or default
    const current = appState.history.length > 0 ? appState.history[appState.history.length-1].level : 'medium';
    dom.statLevel.textContent = current.charAt(0).toUpperCase() + current.slice(1);

    // Update BKT Topic Mastery Radar Chart
    await updateBKTRadarChart();
}

async function updateBKTRadarChart() {
    const canvas = document.getElementById('bkt-radar-chart');
    if (!canvas) return;

    try {
        const res = await fetch(`${API_BASE}/bkt/mastery/${STUDENT_ID}`);
        const data = await res.json();
        
        const avgDisp = document.getElementById('bkt-avg-display');
        if (avgDisp) avgDisp.textContent = `Avg: ${data.overall_mastery_avg}%`;

        const breakdown = data.mastery_breakdown || {};
        const labels = Object.keys(breakdown);
        const scores = labels.map(t => breakdown[t].mastery_percent);
        const target = labels.map(() => 100);

        if (bktRadarChart) {
            bktRadarChart.data.labels = labels;
            bktRadarChart.data.datasets[0].data = scores;
            bktRadarChart.update();
            return;
        }

        const ctx = canvas.getContext('2d');
        bktRadarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Student Mastery %',
                        data: scores,
                        backgroundColor: 'rgba(99, 102, 241, 0.25)',
                        borderColor: '#6366f1',
                        borderWidth: 2,
                        pointBackgroundColor: '#818cf8',
                        pointBorderColor: '#fff'
                    },
                    {
                        label: 'Target Mastery',
                        data: target,
                        backgroundColor: 'rgba(255, 255, 255, 0.03)',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 10, weight: 'bold' } },
                        ticks: { display: false },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    } catch (e) {
        console.warn("BKT Radar Chart error:", e);
    }
}

function addChartBar(isCorrect, level) {
    const bar = document.createElement('div');
    bar.className = `chart-bar ${isCorrect ? '' : 'wrong'}`;
    
    // Map level to height %
    const heightMap = { 'easy': '30%', 'medium': '60%', 'hard': '100%' };
    bar.style.height = heightMap[level] || '50%';
    
    dom.chartContainer.appendChild(bar);
    
    // Limit bars
    if (dom.chartContainer.children.length > 15) {
        dom.chartContainer.removeChild(dom.chartContainer.firstChild);
    }
}

dom.btnNext.addEventListener('click', () => {
    appState.index++;
    if (appState.index < appState.quiz.length) {
        renderQuestion();
    } else {
        finish();
    }
});

function finish() {
    dom.progressBar.style.width = '100%';
    dom.quizRunner.classList.add('hidden');
    dom.resultsScreen.classList.remove('hidden');
    dom.progressPanel.classList.add('hidden'); // Hide partial tracker to focus on final
    
    // Populate Final Stats
    document.getElementById('final-score').textContent = `${appState.correctCount} / ${appState.attempts}`;
    const acc = appState.attempts > 0 ? Math.round((appState.correctCount / appState.attempts) * 100) : 0;
    document.getElementById('final-accuracy').textContent = `${acc}%`;
    const lastLevel = appState.history.length > 0 ? appState.history[appState.history.length-1].level : 'medium';
    document.getElementById('final-level').textContent = lastLevel.charAt(0).toUpperCase() + lastLevel.slice(1);
    
    confetti({ particleCount: 150, spread: 100, origin: { y: 0.5 } });
    lucide.createIcons();
}

dom.btnReset.addEventListener('click', async () => {
    if(!confirm("Are you sure? This will wipe all generated questions and progress.")) return;
    try {
        await fetch(`${API_BASE}/reset-db`, { method: 'DELETE' });
        location.reload();
    } catch(e) { console.error(e); }
});

// --- Tab Switching Navigation ---
function switchTab(tabName) {
    ['quiz', 'chat', 'flashcards'].forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        const sec = document.getElementById(`section-${t}`);
        if (btn) btn.classList.toggle('active', t === tabName);
        if (sec) sec.classList.toggle('hidden', t !== tabName);
    });
    lucide.createIcons();
}

// --- Mermaid.js Diagram Renderer Helper ---
function renderMermaidDiagramsInElement(element) {
    if (!window.mermaid) return;
    const text = element.innerHTML;
    if (text.includes('```mermaid')) {
        const regex = /```mermaid([\s\S]*?)```/g;
        let idCounter = 0;
        const newHtml = text.replace(regex, (match, mermaidCode) => {
            idCounter++;
            const divId = `mermaid-diag-${Date.now()}-${idCounter}`;
            const cleanCode = mermaidCode.replace(/<br>/g, '\n').trim();
            setTimeout(() => {
                try {
                    mermaid.render(divId + '-svg', cleanCode).then(({ svg }) => {
                        const target = document.getElementById(divId);
                        if (target) target.innerHTML = svg;
                    }).catch(err => {
                        console.error("Mermaid Render Error:", err);
                        const target = document.getElementById(divId);
                        if (target) target.innerHTML = `<pre style="text-align:left; font-size:0.75rem; background:#f1f5f9; padding:0.5rem;">${cleanCode}</pre>`;
                    });
                } catch (err) {
                    console.error("Mermaid Exception:", err);
                }
            }, 50);
            return `<div id="${divId}" class="mermaid-container" style="background: #ffffff; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0; margin: 1rem 0; overflow-x: auto; text-align: center;">Rendering Visual Diagram...</div>`;
        });
        element.innerHTML = newHtml;
    }
}

// --- Grounded AI Tutor & Vision Chat Handler ---
const btnSendChat = document.getElementById('btn-send-chat');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const btnUploadImage = document.getElementById('btn-upload-image');
const tutorImageInput = document.getElementById('tutor-image-input');
const attachedImageContainer = document.getElementById('attached-image-container');
const attachedFilename = document.getElementById('attached-filename');
const btnRemoveImage = document.getElementById('btn-remove-image');

if (btnUploadImage && tutorImageInput) {
    btnUploadImage.addEventListener('click', () => tutorImageInput.click());
    tutorImageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            attachedFilename.textContent = e.target.files[0].name;
            attachedImageContainer.classList.remove('hidden');
        }
    });
    if (btnRemoveImage) {
        btnRemoveImage.addEventListener('click', () => {
            tutorImageInput.value = '';
            attachedImageContainer.classList.add('hidden');
        });
    }
}

if (btnSendChat && chatInput) {
    async function sendChatMessage() {
        const query = chatInput.value.trim();
        const imageFile = tutorImageInput && tutorImageInput.files.length ? tutorImageInput.files[0] : null;

        if (!query && !imageFile) return;

        // Append Student Message
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-bubble user';
        userMsg.style.cssText = 'background: var(--primary); color: white; padding: 0.9rem 1.25rem; border-radius: 12px; max-width: 85%; align-self: flex-end; font-weight: 500;';

        if (imageFile) {
            userMsg.innerHTML = `<div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; margin-bottom: 4px; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 8px;">📷 Image Attached: ${imageFile.name}</div><div>${query || 'Analyze uploaded diagram / formula'}</div>`;
        } else {
            userMsg.textContent = query;
        }

        chatMessages.appendChild(userMsg);
        chatInput.value = '';

        if (attachedImageContainer) attachedImageContainer.classList.add('hidden');
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Append Loading Indicator
        const loadingMsg = document.createElement('div');
        loadingMsg.className = 'chat-bubble ai loading';
        loadingMsg.style.cssText = 'background: #f1f5f9; padding: 0.9rem 1.25rem; border-radius: 12px; max-width: 85%; align-self: flex-start; color: var(--text-dim); font-style: italic;';
        loadingMsg.textContent = imageFile ? '📷 Processing Multimodal Vision OCR & Generating Diagram...' : 'Searching course materials & verifying citations...';
        chatMessages.appendChild(loadingMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            let res, data;
            if (imageFile) {
                const formData = new FormData();
                formData.append('file', imageFile);
                if (query) formData.append('query', query);
                formData.append('student_id', STUDENT_ID);

                res = await fetch(`${API_BASE}/chat/vision`, {
                    method: 'POST',
                    body: formData
                });
                if (tutorImageInput) tutorImageInput.value = '';
            } else {
                res = await fetch(`${API_BASE}/chat/tutor`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        student_id: STUDENT_ID,
                        query: query,
                        source_id: appState.sourceId
                    })
                });
            }

            data = await res.json();
            if (chatMessages.contains(loadingMsg)) chatMessages.removeChild(loadingMsg);

            const aiMsg = document.createElement('div');
            aiMsg.className = 'chat-bubble ai';

            if (!data.is_grounded) {
                aiMsg.style.cssText = 'background: #fee2e2; border-left: 4px solid #ef4444; padding: 1rem; border-radius: 12px; max-width: 85%; align-self: flex-start; color: #991b1b;';
                aiMsg.innerHTML = `<strong>⚠️ OUT-OF-MATERIAL REFUSAL</strong><br><br>${data.answer}`;
            } else {
                aiMsg.style.cssText = 'background: #f8fafc; border-left: 4px solid var(--primary); padding: 1rem; border-radius: 12px; max-width: 85%; align-self: flex-start; border: 1px solid var(--border);';

                let citationBadges = (data.citations || []).map(c => 
                    `<span style="background: #dbeafe; color: #1e40af; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: 700; margin-right: 4px;">📌 ${c.citation_label}</span>`
                ).join(' ');

                let formattedAnswer = data.answer.replace(/\n/g, '<br>');
                aiMsg.innerHTML = `<div>${formattedAnswer}</div><div style="margin-top: 0.75rem; font-size: 0.8rem;">${citationBadges}</div>`;
                renderMermaidDiagramsInElement(aiMsg);
            }

            chatMessages.appendChild(aiMsg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (e) {
            console.error("Chat error:", e);
            if (chatMessages.contains(loadingMsg)) chatMessages.removeChild(loadingMsg);
            const errMsg = document.createElement('div');
            errMsg.style.cssText = 'background: #fee2e2; color: #991b1b; padding: 0.75rem; border-radius: 8px;';
            errMsg.textContent = 'Server connection error.';
            chatMessages.appendChild(errMsg);
        }
    }

    btnSendChat.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendChatMessage(); });
}


// --- Revision Flashcards Handler ---
const btnLoadFlashcards = document.getElementById('btn-load-flashcards');
const flashcardGrid = document.getElementById('flashcard-grid');

if (btnLoadFlashcards && flashcardGrid) {
    btnLoadFlashcards.addEventListener('click', async () => {
        try {
            btnLoadFlashcards.disabled = true;
            flashcardGrid.innerHTML = `<div style="color: var(--text-dim);">Fetching targeted flashcards for weak BKT topics...</div>`;
            
            const res = await fetch(`${API_BASE}/revision/flashcards`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ student_id: STUDENT_ID })
            });
            const cards = await res.json();
            
            if (!cards || cards.length === 0) {
                flashcardGrid.innerHTML = `<div style="color: var(--text-dim);">No revision flashcards generated yet. Take a quiz first!</div>`;
                return;
            }

            flashcardGrid.innerHTML = '';
            cards.forEach(c => {
                const cardEl = document.createElement('div');
                cardEl.style.cssText = 'background: #ffffff; border: 1.5px solid var(--border); border-radius: 16px; padding: 1.5rem; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; display: flex; flex-direction: column; justify-content: space-between; min-height: 200px; box-shadow: 0 4px 12px rgba(15,23,42,0.04);';
                cardEl.innerHTML = `
                    <div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                            <span style="font-size: 0.75rem; font-weight: 700; color: var(--primary); text-transform: uppercase;">${c.topic}</span>
                            <span style="font-size: 0.75rem; color: #64748b; font-family: monospace;">${c.source_citation}</span>
                        </div>
                        <h4 style="font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-main);">${c.concept_title}</h4>
                        <p style="font-size: 0.9rem; color: #475569; font-weight: 500;" class="card-text">${c.front_prompt}</p>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--primary); font-weight: 700; margin-top: 1rem; text-align: right;">💡 Click to reveal answer</div>
                `;
                
                let isFlipped = false;
                cardEl.addEventListener('click', () => {
                    const p = cardEl.querySelector('.card-text');
                    if (isFlipped) {
                        p.textContent = c.front_prompt;
                        p.style.color = '#475569';
                        isFlipped = false;
                    } else {
                        p.textContent = c.back_explanation;
                        p.style.color = '#047857';
                        isFlipped = true;
                    }
                });

                flashcardGrid.appendChild(cardEl);
            });
        } catch (e) {
            flashcardGrid.innerHTML = `<div style="color: var(--danger);">Failed to load flashcards.</div>`;
        } finally {
            btnLoadFlashcards.disabled = false;
        }
    });
}


