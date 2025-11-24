// Global state
let currentStyleName = '';

// Analyze Style
async function analyzeStyle() {
    const stylizedText = document.getElementById('stylizedText').value.trim();
    const styleName = document.getElementById('styleName').value.trim() || 'custom_style';
    
    if (!stylizedText) {
        alert('Please enter stylized text to analyze');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                stylized_text: stylizedText,
                style_name: styleName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentStyleName = data.style_name;
            displayAnalysisResult(data.summary);
            
            // Enable transfer button
            document.getElementById('transferBtn').disabled = false;
        } else {
            alert('Error: ' + (data.error || 'Analysis failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze style. Please check if Ollama is running.');
    } finally {
        hideLoading();
    }
}

// Display Analysis Result
function displayAnalysisResult(summary) {
    const resultDiv = document.getElementById('analysisResult');
    const detailsDiv = document.getElementById('styleDetails');
    
    detailsDiv.innerHTML = `
        <div class="style-metric">
            <div class="style-metric-label">Avg Word Length</div>
            <div class="style-metric-value">${summary.avg_word_length.toFixed(2)} chars</div>
        </div>
        <div class="style-metric">
            <div class="style-metric-label">Sentence Length</div>
            <div class="style-metric-value">${summary.avg_sentence_length.toFixed(1)} words</div>
        </div>
        <div class="style-metric">
            <div class="style-metric-label">Vocabulary Richness</div>
            <div class="style-metric-value">${(summary.vocabulary_richness * 100).toFixed(1)}%</div>
        </div>
        <div class="style-metric">
            <div class="style-metric-label">Formality</div>
            <div class="style-metric-value">${(summary.formality_score * 100).toFixed(0)}%</div>
        </div>
        <div class="style-metric">
            <div class="style-metric-label">Sentiment</div>
            <div class="style-metric-value">${summary.sentiment_polarity > 0 ? '😊' : summary.sentiment_polarity < 0 ? '😔' : '😐'} ${summary.sentiment_polarity.toFixed(2)}</div>
        </div>
        <div class="style-metric">
            <div class="style-metric-label">Reading Level</div>
            <div class="style-metric-value">Grade ${summary.flesch_kincaid_grade.toFixed(1)}</div>
        </div>
    `;
    
    // Show top words
    if (summary.top_words && summary.top_words.length > 0) {
        detailsDiv.innerHTML += `
            <div class="style-metric" style="grid-column: 1 / -1;">
                <div class="style-metric-label">Top Words</div>
                <div class="style-metric-value" style="font-size: 0.9rem;">
                    ${summary.top_words.slice(0, 10).join(', ')}
                </div>
            </div>
        `;
    }
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Transfer Style
async function transferStyle() {
    const originalText = document.getElementById('originalText').value.trim();
    
    if (!originalText) {
        alert('Please enter text to transform');
        return;
    }
    
    if (!currentStyleName) {
        alert('Please analyze a style first');
        return;
    }
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch('/transfer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                original_text: originalText,
                style_name: currentStyleName,
                author_name: document.getElementById('styleName').value.trim() || 'the target style'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayTransferResult(data.original, data.styled);
        } else {
            alert('Error: ' + (data.error || 'Transfer failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to transfer style. Please check if Ollama is running.');
    } finally {
        hideLoading();
    }
}

// Display Transfer Result
function displayTransferResult(original, styled) {
    const resultDiv = document.getElementById('transferResult');
    
    document.getElementById('originalDisplay').textContent = original;
    document.getElementById('styledDisplay').textContent = styled;
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Loading helpers
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to analyze
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement.id === 'stylizedText') {
            analyzeStyle();
        } else if (activeElement.id === 'originalText') {
            transferStyle();
        }
    }
});

// Auto-save to localStorage
const stylizedTextArea = document.getElementById('stylizedText');
const originalTextArea = document.getElementById('originalText');
const styleNameInput = document.getElementById('styleName');

// Load from localStorage on page load
window.addEventListener('load', function() {
    const savedStylized = localStorage.getItem('stylizedText');
    const savedOriginal = localStorage.getItem('originalText');
    const savedStyleName = localStorage.getItem('styleName');
    
    if (savedStylized) stylizedTextArea.value = savedStylized;
    if (savedOriginal) originalTextArea.value = savedOriginal;
    if (savedStyleName) styleNameInput.value = savedStyleName;
});

// Save to localStorage on change
stylizedTextArea.addEventListener('input', function() {
    localStorage.setItem('stylizedText', this.value);
});

originalTextArea.addEventListener('input', function() {
    localStorage.setItem('originalText', this.value);
});

styleNameInput.addEventListener('input', function() {
    localStorage.setItem('styleName', this.value);
});
