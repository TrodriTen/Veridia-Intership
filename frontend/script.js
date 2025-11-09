// Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');
const resumeTextArea = document.getElementById('resume-text');
const charCount = document.getElementById('char-count');
const analyzeTextBtn = document.getElementById('analyze-text-btn');
const fileUploadArea = document.getElementById('file-upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const analyzeFileBtn = document.getElementById('analyze-file-btn');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const resetBtn = document.getElementById('reset-btn');
const errorResetBtn = document.getElementById('error-reset-btn');

// State
let currentFile = null;

// Tab Navigation
tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tabName = button.dataset.tab;
        
        // Update buttons
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Update content
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === `${tabName}-tab`) {
                content.classList.add('active');
            }
        });
        
        // Reset states
        hideResults();
        hideError();
    });
});

// Character Counter
resumeTextArea.addEventListener('input', (e) => {
    const length = e.target.value.length;
    charCount.textContent = length;
    analyzeTextBtn.disabled = length < 100;
});

// File Upload - Click
fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File Upload - Input Change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// File Upload - Drag & Drop
fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('drag-over');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('drag-over');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Handle File Selection
function handleFileSelect(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['text/plain', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a TXT or PDF file.');
        return;
    }
    
    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 10MB.');
        return;
    }
    
    currentFile = file;
    
    // Show file info
    fileUploadArea.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    fileInfo.querySelector('.file-name').textContent = file.name;
    analyzeFileBtn.disabled = false;
}

// Remove File
fileInfo.querySelector('.remove-file-btn').addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    fileUploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analyzeFileBtn.disabled = true;
});

// Analyze Text
analyzeTextBtn.addEventListener('click', async () => {
    const text = resumeTextArea.value.trim();
    
    if (text.length < 100) {
        showError('Resume text is too short. Please provide at least 100 characters.');
        return;
    }
    
    setLoading(analyzeTextBtn, true);
    hideError();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze resume');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while analyzing the resume. Please try again.');
    } finally {
        setLoading(analyzeTextBtn, false);
    }
});

// Analyze File
analyzeFileBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    setLoading(analyzeFileBtn, true);
    hideError();
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/file`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze resume');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while analyzing the file. Please try again.');
    } finally {
        setLoading(analyzeFileBtn, false);
    }
});

// Display Results
function displayResults(result) {
    // Hide input sections
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelector('.tabs').classList.add('hidden');
    
    // Show results
    resultsSection.classList.remove('hidden');
    
    // Display category
    document.getElementById('predicted-category').textContent = result.category;
    
    // Display confidence
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence-value').textContent = `${confidence}%`;
    document.getElementById('confidence-progress').style.width = `${confidence}%`;
    
    // Display probabilities chart
    displayProbabilitiesChart(result.probabilities);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Probabilities Chart
function displayProbabilitiesChart(probabilities) {
    const chartContainer = document.getElementById('probabilities-chart');
    chartContainer.innerHTML = '';
    
    // Sort by probability and get top 5
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    
    sortedProbs.forEach(([category, probability]) => {
        const percentage = (probability * 100).toFixed(1);
        
        const chartItem = document.createElement('div');
        chartItem.className = 'chart-item';
        
        chartItem.innerHTML = `
            <div class="chart-label">${category}</div>
            <div class="chart-bar-container">
                <div class="chart-bar">
                    <div class="chart-bar-fill" style="width: 0%"></div>
                </div>
                <div class="chart-value">${percentage}%</div>
            </div>
        `;
        
        chartContainer.appendChild(chartItem);
        
        // Animate bar
        setTimeout(() => {
            chartItem.querySelector('.chart-bar-fill').style.width = `${percentage}%`;
        }, 100);
    });
}

// Show Error
function showError(message) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.querySelector('.tabs').classList.add('hidden');
    resultsSection.classList.add('hidden');
    
    errorSection.classList.remove('hidden');
    document.getElementById('error-message').textContent = message;
    
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Hide Error
function hideError() {
    errorSection.classList.add('hidden');
}

// Hide Results
function hideResults() {
    resultsSection.classList.add('hidden');
}

// Reset
function resetApp() {
    // Reset text input
    resumeTextArea.value = '';
    charCount.textContent = '0';
    analyzeTextBtn.disabled = true;
    
    // Reset file input
    currentFile = null;
    fileInput.value = '';
    fileUploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analyzeFileBtn.disabled = true;
    
    // Show tabs and first tab
    document.querySelector('.tabs').classList.remove('hidden');
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('hidden', 'active'));
    document.getElementById('text-tab').classList.add('active');
    
    // Reset tab buttons
    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabButtons[0].classList.add('active');
    
    // Hide results and errors
    hideResults();
    hideError();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

resetBtn.addEventListener('click', resetApp);
errorResetBtn.addEventListener('click', resetApp);

// Loading State
function setLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.querySelector('.btn-text').classList.add('hidden');
        button.querySelector('.btn-loader').classList.remove('hidden');
    } else {
        button.disabled = false;
        button.querySelector('.btn-text').classList.remove('hidden');
        button.querySelector('.btn-loader').classList.add('hidden');
    }
}

// Check API Health on Load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.models_loaded) {
            showError('API is running but models are not loaded. Please train the model first by running: ./train.sh');
        }
    } catch (error) {
        console.warn('Could not connect to API:', error);
        // Don't show error on initial load, might be loading
    }
}

// Initialize
checkAPIHealth();
