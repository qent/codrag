let pollTimer = null;

// Theme handling (auto | light | dark)
const THEME_KEY = 'theme';
function systemTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
function savedTheme() {
    const t = localStorage.getItem(THEME_KEY);
    if (t === 'light' || t === 'dark' || t === 'auto') return t;
    return 'auto';
}
function effectiveTheme() {
    const s = savedTheme();
    return s === 'auto' ? systemTheme() : s;
}
function applyTheme(theme, persist = false) {
    if (theme === 'light' || theme === 'dark') {
        document.documentElement.setAttribute('data-theme', theme);
    } else {
        // auto: remove override to follow system
        document.documentElement.removeAttribute('data-theme');
        theme = 'auto';
    }
    if (persist) localStorage.setItem(THEME_KEY, theme);
    renderThemeUI();
}
function renderThemeUI() {
    const mode = savedTheme();
    const eff = effectiveTheme();
    const autoBtn = document.getElementById('themeAuto');
    const lightBtn = document.getElementById('themeLight');
    const darkBtn = document.getElementById('themeDark');
    const setActive = (btn, active) => {
        if (!btn) return;
        btn.classList.toggle('active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    };
    setActive(autoBtn, mode === 'auto');
    setActive(lightBtn, mode === 'light');
    setActive(darkBtn, mode === 'dark');
}
(function initTheme() {
    // Apply saved theme (auto by default)
    const saved = savedTheme();
    applyTheme(saved, false);
    // Attach listeners
    const autoBtn = document.getElementById('themeAuto');
    const lightBtn = document.getElementById('themeLight');
    const darkBtn = document.getElementById('themeDark');
    if (autoBtn) autoBtn.addEventListener('click', () => applyTheme('auto', true));
    if (lightBtn) lightBtn.addEventListener('click', () => applyTheme('light', true));
    if (darkBtn) darkBtn.addEventListener('click', () => applyTheme('dark', true));
    // Update if system theme changes and mode is auto
    const mq = window.matchMedia('(prefers-color-scheme: light)');
    if (mq && mq.addEventListener) {
        mq.addEventListener('change', () => {
            if (savedTheme() === 'auto') applyTheme('auto', false);
        });
    }
})();

function setInProgress(inProgress, elapsedMs = 0) {
    const submit = document.getElementById('submit');
    const cancel = document.getElementById('cancel');
    const statusText = document.getElementById('statusText');
    const dot = document.getElementById('dot');
    const result = document.getElementById('result');
    submit.disabled = inProgress;
    cancel.disabled = !inProgress;
    result.setAttribute('aria-busy', inProgress ? 'true' : 'false');
    if (inProgress) {
        const secs = Math.floor(elapsedMs / 1000);
        const mm = String(Math.floor(secs / 60)).padStart(2, '0');
        const ss = String(secs % 60).padStart(2, '0');
        statusText.textContent = `Working… ${mm}:${ss}`;
        dot.style.display = 'inline-block';
    } else {
        statusText.textContent = '';
        dot.style.display = 'none';
    }
}

async function pollStatus() {
    try {
        const r = await fetch('/search/status');
        const s = await r.json();
        if (s.in_progress) {
            setInProgress(true, s.elapsed_ms || 0);
            return;
        }
        setInProgress(false);
        clearInterval(pollTimer); pollTimer = null;
        const out = document.getElementById('result');
        // Stream HTML via SSE and render progressively
        const es = new EventSource('/search/result/stream');
        await new Promise((resolve) => {
            es.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data && data.html) {
                        out.innerHTML = data.html;
                    } else if (data && data.error) {
                        out.textContent = data.error;
                    }
                } catch (_) {
                    // Ignore malformed frames
                }
            };
            es.onerror = () => {
                // Connection closed (normally or due to error)
                try { es.close(); } catch (_) {}
                resolve();
            };
        });
        // Fetch the final result to ensure we show the completed render
        try {
            const finalRes = await fetch('/search/result');
            const finalData = await finalRes.json();
            if (finalData && finalData.html) {
                out.innerHTML = finalData.html;
            }
        } catch (_) {}
    } catch (_) {
        // ignore transient polling errors
    }
}

async function beginPolling() {
    if (pollTimer) return;
    await pollStatus();
    pollTimer = setInterval(pollStatus, 1200);
}

async function startSearch() {
    const q = document.getElementById('query').value;
    const out = document.getElementById('result');
    out.textContent = '';
    const res = await fetch('/search/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
    });
    if (res.status === 409) {
        setInProgress(true);
        await beginPolling();
        return;
    }
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        out.textContent = JSON.stringify(err, null, 2);
        return;
    }
    setInProgress(true);
    await beginPolling();
}

document.getElementById('submit').onclick = startSearch;
document.getElementById('cancel').onclick = async () => {
    await fetch('/search/cancel', { method: 'POST' });
    setTimeout(pollStatus, 300);
};

// Shortcuts: Ctrl/⌘ + Enter to submit
const queryEl = document.getElementById('query');
queryEl.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        startSearch();
    }
});

// On load, reflect current state
(async () => {
    const r = await fetch('/search/status');
    const s = await r.json();
    if (s.in_progress) {
        setInProgress(true, s.elapsed_ms || 0);
        await beginPolling();
    }
})();
