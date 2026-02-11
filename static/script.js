/* ============================================================================
   INITIALIZATION & UTILITIES
   ============================================================================ */

// Lab preset values
const LAB_PRESETS = {
    normal: { crp: 1.0, wbc: 7.0, hb: 14.0 },
    tumor: { crp: 10.0, wbc: 9.0, hb: 11.0 },
    infection: { crp: 50.0, wbc: 15.0, hb: 12.0 },
    inflammatory: { crp: 20.0, wbc: 8.0, hb: 14.0 }
};

const CLASS_COLORS = {
    'Normal': '#4CAF50',
    'Tumor': '#f44336',
    'Infection': '#FF9800',
    'Inflammatory': '#FFC107'
};

let selectedImageFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializePage();
    setupEventListeners();
    checkModelStatus();
});

function initializePage() {
    setupTabNavigation();
    setupDragDrop();
    setupImageInput();
}

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchTab(e.target.dataset.tab);
        });
    });
}

function setupTabNavigation() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active to clicked tab
            this.classList.add('active');
            const tabId = this.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
}

/* ============================================================================
   DRAG & DROP
   ============================================================================ */

function setupDragDrop() {
    const dropZone = document.getElementById('dropZone');
    
    dropZone.addEventListener('click', () => {
        document.getElementById('imageInput').click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageSelect(files[0]);
        }
    });
}

function setupImageInput() {
    const imageInput = document.getElementById('imageInput');
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageSelect(e.target.files[0]);
        }
    });
}

function handleImageSelect(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom', 'application/dicom+json'];
    const validExtensions = ['.png', '.jpg', '.jpeg', '.dcm'];
    const fileExt = file.name.split('.').pop().toLowerCase();
    const isDicom = fileExt === 'dcm';
    const isValidType = validTypes.includes(file.type) || isDicom;

    if (!isValidType) {
        showError('Invalid file type. Please upload PNG, JPG, or DCM.');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Maximum size is 50MB.');
        return;
    }

    selectedImageFile = file;

    if (isDicom) {
        // No preview for DICOM, just show filename
        const preview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');
        previewImg.src = '';
        previewImg.alt = file.name + ' (DICOM)';
        preview.style.display = 'block';
        document.getElementById('dropZone').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    } else {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('imagePreview');
            const previewImg = document.getElementById('previewImg');
            previewImg.src = e.target.result;
            preview.style.display = 'block';
            document.getElementById('dropZone').style.display = 'none';
            document.getElementById('predictBtn').disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

function clearImage() {
    selectedImageFile = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('predictBtn').disabled = true;
}

/* ============================================================================
   LAB VALUES
   ============================================================================ */

function setLabPreset(preset) {
    const values = LAB_PRESETS[preset];
    document.getElementById('crp').value = values.crp;
    document.getElementById('wbc').value = values.wbc;
    document.getElementById('hb').value = values.hb;
}

/* ============================================================================
   PREDICTION
   ============================================================================ */

async function predictImage() {
    if (!selectedImageFile) {
        showError('Please select an image first.');
        return;
    }

    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('image', selectedImageFile);
        formData.append('crp', document.getElementById('crp').value);
        formData.append('wbc', document.getElementById('wbc').value);
        formData.append('hb', document.getElementById('hb').value);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            showError(error.error || 'Prediction failed');
            return;
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Update prediction
    document.getElementById('resultClass').textContent = result.predicted_class;
    document.getElementById('resultConfidence').textContent = 
        `${(result.confidence * 100).toFixed(2)}% confident`;
    
    // Update confidence bar
    const confidencePercent = result.confidence * 100;
    document.getElementById('confidenceBarFill').style.width = confidencePercent + '%';
    document.getElementById('confidencePercent').textContent = 
        confidencePercent.toFixed(1) + '%';
    
    // Update probabilities
    const probContainer = document.getElementById('probabilitiesContainer');
    probContainer.innerHTML = '';
    
    Object.entries(result.probabilities).forEach(([className, prob]) => {
        const percent = (prob * 100).toFixed(1);
        let levelClass = 'low';
        if (prob > 0.7) levelClass = 'high';
        else if (prob > 0.4) levelClass = 'medium';
        
        const card = document.createElement('div');
        card.className = `probability-card ${levelClass}`;
        card.innerHTML = `
            <span class="prob-label">${className}</span>
            <span class="prob-value">${percent}%</span>
        `;
        probContainer.appendChild(card);
    });
    
    // Update explanation
    document.getElementById('explanation').textContent = result.explanation;
    
    // Update lab summary
    document.getElementById('summarycrp').textContent = result.lab_values.crp.toFixed(2);
    document.getElementById('summarywbc').textContent = result.lab_values.wbc.toFixed(2);
    document.getElementById('summaryhb').textContent = result.lab_values.hb.toFixed(2);
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/* ============================================================================
   DEMO MODE
   ============================================================================ */

async function runDemo(classIndex) {
    showLoading(true);

    try {
        const response = await fetch('/api/predict-demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ disease_class: classIndex })
        });

        if (!response.ok) {
            const error = await response.json();
            showError(error.error || 'Demo failed');
            return;
        }

        const result = await response.json();
        displayDemoResults(result);
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function displayDemoResults(result) {
    const resultsSection = document.getElementById('demoResultsSection');
    
    // Update results
    document.getElementById('demoTrueClass').textContent = result.true_class;
    document.getElementById('demoPredictedClass').textContent = result.predicted_class;
    document.getElementById('demoConfidence').textContent = 
        `${(result.confidence * 100).toFixed(2)}%`;
    
    // Update probabilities
    const probContainer = document.getElementById('demoProbabilities');
    probContainer.innerHTML = '';
    
    Object.entries(result.probabilities).forEach(([className, prob]) => {
        const percent = (prob * 100).toFixed(1);
        let levelClass = 'low';
        if (prob > 0.7) levelClass = 'high';
        else if (prob > 0.4) levelClass = 'medium';
        
        const card = document.createElement('div');
        card.className = `probability-card ${levelClass}`;
        card.innerHTML = `
            <span class="prob-label">${className}</span>
            <span class="prob-value">${percent}%</span>
        `;
        probContainer.appendChild(card);
    });
    
    // Update explanation
    document.getElementById('demoExplanation').textContent = result.explanation;
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/* ============================================================================
   MODEL STATUS
   ============================================================================ */

async function checkModelStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusDot = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        if (data.model_loaded) {
            statusDot.classList.add('ready');
            statusText.textContent = `Ready (${data.device})`;
        } else {
            statusText.textContent = 'Model not loaded';
        }
        
        // Display GPU stats if available
        if (data.gpu_available && data.gpu_info) {
            const gpuStats = document.getElementById('gpu-stats');
            document.getElementById('gpu-device').textContent = `GPU: ${data.gpu_info.name}`;
            document.getElementById('gpu-memory').textContent = `Memory: ${data.gpu_info.used_gb}/${data.gpu_info.memory_gb} GB`;
            gpuStats.style.display = 'block';
        }
    } catch (error) {
        document.getElementById('status-text').textContent = 'Connection error';
        console.error('Health check failed:', error);
    }
}

/* ============================================================================
   UI UTILITIES
   ============================================================================ */

function showLoading(show) {
    const loader = document.getElementById('loadingIndicator');
    if (show) {
        loader.classList.add('active');
    } else {
        loader.classList.remove('active');
    }
}

function showError(message) {
    const toast = document.getElementById('errorToast');
    document.getElementById('errorMessage').textContent = message;
    toast.style.display = 'flex';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 5000);
}

function closeToast() {
    document.getElementById('errorToast').style.display = 'none';
}
