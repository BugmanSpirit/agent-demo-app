import './style.css'

// State
let currentSystemPrompt = '';
let abortController = null;
let currentMode = 'text';
let autoScroll = true;

// DOM Elements
const searchInterface = document.getElementById('search-interface');
const resultsInterface = document.getElementById('results-interface');
const loadingState = document.getElementById('loading-state');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const newSearchBtn = document.getElementById('new-search-btn');
const resultsContent = document.getElementById('results-content');
const configModal = document.getElementById('config-modal');
const configBtn = document.getElementById('config-btn');
const systemPromptTextarea = document.getElementById('system-prompt');
const saveConfigBtn = document.getElementById('save-config');
const cancelConfigBtn = document.getElementById('cancel-config');
const themeToggle = document.getElementById('theme-toggle');

// Configure marked
marked.setOptions({
    breaks: true,
    gfm: true
});

// Theme Management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.classList.toggle('dark', savedTheme === 'dark');
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Auto-scroll to bottom
function scrollToBottom() {
    if (autoScroll) {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Event Listeners
searchBtn.addEventListener('click', startResearch);
newSearchBtn.addEventListener('click', showSearchInterface);
configBtn.addEventListener('click', showConfigModal);
saveConfigBtn.addEventListener('click', saveConfig);
cancelConfigBtn.addEventListener('click', hideConfigModal);
themeToggle.addEventListener('click', toggleTheme);

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') startResearch();
});

// Example queries
document.querySelectorAll('.example-query').forEach(btn => {
    btn.addEventListener('click', () => {
        searchInput.value = btn.textContent.trim();
        startResearch();
    });
});

// Functions
function showSearchInterface() {
    searchInterface.classList.remove('hidden');
    resultsInterface.classList.add('hidden');
    loadingState.classList.add('hidden');
    resultsContent.innerHTML = '';
    searchInput.value = '';
    currentMode = 'text';
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showLoadingState() {
    searchInterface.classList.add('hidden');
    resultsInterface.classList.add('hidden');
    loadingState.classList.remove('hidden');
}

function showResults() {
    searchInterface.classList.add('hidden');
    resultsInterface.classList.remove('hidden');
    loadingState.classList.add('hidden');
}

function showConfigModal() {
    systemPromptTextarea.value = currentSystemPrompt;
    configModal.classList.remove('hidden');
}

function hideConfigModal() {
    configModal.classList.add('hidden');
}

function saveConfig() {
    currentSystemPrompt = systemPromptTextarea.value;
    hideConfigModal();
}

async function startResearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    showLoadingState();
    currentMode = 'text';
    autoScroll = true;

    if (abortController) {
        abortController.abort();
    }

    abortController = new AbortController();

    try {
        const response = await fetch('/agents/cerebras-search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                systemPrompt: currentSystemPrompt || undefined,
            }),
            signal: abortController.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
            throw new Error('No response body');
        }

        let buffer = '';
        showResults();

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        return;
                    }

                    try {
                        const chunk = JSON.parse(data);
                        handleStreamChunk(chunk);
                        scrollToBottom();
                    } catch (error) {
                        console.error('Error parsing chunk:', error, data);
                    }
                }
            }
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Request was aborted');
        } else {
            console.error('Research error:', error);
            showError(`Research failed: ${error.message}`);
        }
    } finally {
        abortController = null;
    }
}

function handleStreamChunk(chunk) {
    switch (chunk.type) {
        case 'text-delta':
            if (currentMode === 'reasoning') {
                finalizeCurrentSection();
                currentMode = 'text';
            }
            appendText(chunk.text || '');
            break;
        case 'reasoning-delta':
            if (currentMode === 'text') {
                finalizeCurrentSection();
                currentMode = 'reasoning';
            }
            appendReasoning(chunk.text || '');
            break;
        case 'tool-call':
            finalizeCurrentSection();
            addToolCall(chunk);
            break;
        case 'tool-result':
            addToolResult(chunk);
            break;
        case 'error':
            showError(chunk.error?.message || chunk.error?.lastError?.data?.message);
            break;
        case 'finish':
            finalizeCurrentSection();
            addFinishIndicator(chunk.finishReason);
            console.log('Research completed with reason:', chunk.finishReason);
            break;
    }
}

function appendText(text) {
    if (!text) return;

    let currentElement = resultsContent.querySelector('.current-text');
    if (!currentElement) {
        currentElement = document.createElement('div');
        currentElement.className = 'current-text bg-card border border-border rounded-xl p-6 animate-fade-in';

        const markdownContainer = document.createElement('div');
        markdownContainer.className = 'markdown-content prose dark:prose-invert max-w-none';
        currentElement.appendChild(markdownContainer);

        resultsContent.appendChild(currentElement);
    }

    if (!currentElement.rawText) {
        currentElement.rawText = '';
    }
    currentElement.rawText += text;

    const textContainer = currentElement.querySelector('.markdown-content');
    if (textContainer) {
        textContainer.innerHTML = marked.parse(currentElement.rawText);
    }
}

function appendReasoning(text) {
    if (!text) return;

    let currentElement = resultsContent.querySelector('.current-reasoning');
    if (!currentElement) {
        currentElement = document.createElement('div');
        currentElement.className = 'current-reasoning bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-l-4 border-blue-500 dark:border-blue-400 p-4 rounded-lg animate-fade-in';

        const header = document.createElement('div');
        header.className = 'flex items-center gap-2 mb-2';
        header.innerHTML = `
            <div class="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full animate-pulse"></div>
            <span class="font-medium text-sm text-blue-700 dark:text-blue-300">AI Reasoning</span>
        `;
        currentElement.appendChild(header);

        const markdownContainer = document.createElement('div');
        markdownContainer.className = 'markdown-content prose dark:prose-invert prose-sm max-w-none';
        currentElement.appendChild(markdownContainer);

        resultsContent.appendChild(currentElement);
    }

    if (!currentElement.rawText) {
        currentElement.rawText = '';
    }
    currentElement.rawText += text;

    const textContainer = currentElement.querySelector('.markdown-content');
    if (textContainer) {
        textContainer.innerHTML = marked.parse(currentElement.rawText);
    }
}

function finalizeCurrentSection() {
    const currentTextElement = resultsContent.querySelector('.current-text');
    if (currentTextElement) {
        currentTextElement.classList.remove('current-text');
    }

    const currentReasoningElement = resultsContent.querySelector('.current-reasoning');
    if (currentReasoningElement) {
        currentReasoningElement.classList.remove('current-reasoning');
    }
}

function addToolCall(toolCall) {
    finalizeCurrentSection();

    const toolCallElement = document.createElement('div');
    toolCallElement.className = 'bg-gradient-to-r from-orange-500/10 to-red-500/10 border-l-4 border-orange-500 dark:border-orange-400 p-4 rounded-lg animate-fade-in';

    const objective = toolCall.args?.objective || toolCall.input?.objective || 'Searching...';

    toolCallElement.innerHTML = `
        <div class="flex items-center gap-2 mb-2">
            <div class="w-2 h-2 bg-orange-500 dark:bg-orange-400 rounded-full animate-pulse"></div>
            <span class="font-medium text-sm">Searching Web</span>
        </div>
        <div class="text-sm text-muted-foreground">
            <div><strong>Objective:</strong> ${escapeHtml(objective)}</div>
        </div>
    `;
    resultsContent.appendChild(toolCallElement);
}

function addToolResult(toolResult) {
    const results = toolResult?.output?.results;
    if (!results) return;

    const resultElement = document.createElement('div');
    resultElement.className = 'bg-card border border-border rounded-xl p-6 animate-fade-in';

    const totalExcerpts = results.slice(0, 5).reduce((sum, result) => sum + (result.excerpts?.length || 0), 0);

    let resultsHtml = `
        <div class="flex items-center justify-between mb-4">
            <div class="flex items-center gap-2 font-medium text-sm">
                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                Search Results Found
            </div>
            ${totalExcerpts > 0 ? `
                <button class="excerpts-toggle-btn text-xs text-primary-500 hover:text-primary-600 underline" data-expanded="false">
                    Show Excerpts (${totalExcerpts})
                </button>
            ` : ''}
        </div>
    `;

    if (results && Array.isArray(results)) {
        results.slice(0, 5).forEach((result, index) => {
            const hasExcerpts = result.excerpts && result.excerpts.length > 0;
            const urlWithHighlights = createHighlightUrl(result.url, result.excerpts || []);

            resultsHtml += `
                <div class="mb-4 pb-4 ${index < 4 ? 'border-b border-border' : ''}" data-result-index="${index}">
                    <div class="font-medium text-primary-500 mb-1">
                        <a href="${urlWithHighlights}" target="_blank" class="hover:underline text-sm">${escapeHtml(result.title)}</a>
                    </div>
                    <div class="text-xs text-muted-foreground truncate mb-2">${escapeHtml(result.url)}</div>
                    ${hasExcerpts ? `
                        <div class="excerpts-container hidden">
                            <div class="text-xs font-medium text-muted-foreground mb-2">Excerpts:</div>
                            <div class="space-y-2">
                                ${result.excerpts.map(excerpt => `
                                    <div class="text-xs text-muted-foreground bg-accent/50 p-3 rounded-lg border-l-2 border-primary-500 whitespace-pre-wrap break-words">
                                        ${escapeHtml(excerpt)}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        });

        if (results.length > 5) {
            resultsHtml += `<div class="text-xs text-muted-foreground text-center pt-2">+ ${results.length - 5} more results</div>`;
        }
    }

    resultElement.innerHTML = resultsHtml;

    const toggleBtn = resultElement.querySelector('.excerpts-toggle-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            const isExpanded = this.dataset.expanded === 'true';
            const excerptContainers = resultElement.querySelectorAll('.excerpts-container');

            if (isExpanded) {
                excerptContainers.forEach(container => container.classList.add('hidden'));
                this.textContent = `Show Excerpts (${totalExcerpts})`;
                this.dataset.expanded = 'false';
            } else {
                excerptContainers.forEach(container => container.classList.remove('hidden'));
                this.textContent = `Hide Excerpts`;
                this.dataset.expanded = 'true';
            }
        });
    }

    resultsContent.appendChild(resultElement);
}

function createHighlightUrl(originalUrl, excerpts) {
    if (!excerpts || !Array.isArray(excerpts) || excerpts.length === 0) {
        return originalUrl;
    }

    const textFragments = excerpts
        .map(excerpt => {
            if (typeof excerpt !== 'string') return '';
            return excerpt.replace(/\.{2,}$/, '').trim();
        })
        .filter(excerpt => excerpt.length > 0)
        .map(encodeURIComponent);

    if (textFragments.length === 0) {
        return originalUrl;
    }

    const fragmentParam = textFragments.map(fragment => `text=${fragment}`).join('&');
    const separator = originalUrl.includes('#') ? '&' : '#:~:';
    return `${originalUrl}${separator}${fragmentParam}`;
}

function addFinishIndicator(finishReason) {
    const finishElement = document.createElement('div');
    finishElement.className = 'bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-l-4 border-green-500 dark:border-green-400 p-4 rounded-lg animate-fade-in';

    const reasonText = finishReason || 'completed';
    const reasonDisplay = reasonText.charAt(0).toUpperCase() + reasonText.slice(1);

    finishElement.innerHTML = `
        <div class="flex items-center gap-2">
            <div class="w-2 h-2 bg-green-500 dark:bg-green-400 rounded-full"></div>
            <span class="font-medium text-sm text-green-700 dark:text-green-300">Research Complete</span>
            <span class="ml-2 text-xs text-green-600 dark:text-green-400">• ${reasonDisplay}</span>
        </div>
    `;
    resultsContent.appendChild(finishElement);
}

function showError(error) {
    showResults();
    finalizeCurrentSection();

    const errorElement = document.createElement('div');
    errorElement.className = 'bg-red-50Built with Parallel AI & Google Gemini • 0/10 border-l-4 border-red-500 dark:border-red-400 p-4 rounded-lg animate-fade-in';
    errorElement.innerHTML = `
        <div class="flex items-center gap-2 text-sm">
            <div class="w-2 h-2 bg-red-500 dark:bg-red-400 rounded-full"></div>
            <strong class="text-red-700 dark:text-red-300">Error:</strong>
            <span class="text-red-600 dark:text-red-400">${escapeHtml(error)}</span>
        </div>
    `;
    resultsContent.appendChild(errorElement);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
initTheme();
showSearchInterface();
