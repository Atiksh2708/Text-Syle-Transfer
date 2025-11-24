"""
Flask Web Application for Style Transfer
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from infer import StyleAnalyzer, StyleTransferEngine, StyleVector
from dataclasses import asdict

app = Flask(__name__)

# Initialize analyzer and engine
analyzer = StyleAnalyzer()
engine = StyleTransferEngine()

# Store style vectors in memory (could be moved to database)
style_cache = {}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_style():
    """Analyze stylized text and create style vector"""
    try:
        data = request.json
        stylized_text = data.get('stylized_text', '')
        style_name = data.get('style_name', 'custom_style')
        
        if not stylized_text:
            return jsonify({'error': 'No stylized text provided'}), 400
        
        print(f"\n[ANALYZE] Received request for style: {style_name}")
        print(f"[ANALYZE] Text length: {len(stylized_text)} characters")
        
        # Split into sentences
        texts = [line.strip() for line in stylized_text.split('\n') if line.strip()]
        
        if len(texts) < 5:
            # If too few lines, split by sentences
            import re
            texts = [s.strip() for s in re.split(r'[.!?]+', stylized_text) if s.strip()]
        
        print(f"[ANALYZE] Processing {len(texts)} text segments")
        
        # Analyze the style
        style_vector = analyzer.analyze_corpus(texts)
        
        print(f"[ANALYZE] Analysis complete for: {style_name}")
        
        # Cache the style vector
        style_cache[style_name] = style_vector
        
        # Return summary
        return jsonify({
            'success': True,
            'style_name': style_name,
            'summary': {
                'avg_word_length': round(style_vector.avg_word_length, 2),
                'avg_sentence_length': round(style_vector.avg_words_per_sentence, 2),
                'vocabulary_richness': round(style_vector.vocabulary_richness, 3),
                'formality_score': round(style_vector.formality_score, 3),
                'sentiment_polarity': round(style_vector.sentiment_polarity, 3),
                'flesch_kincaid_grade': round(style_vector.flesch_kincaid_grade, 1),
                'top_words': style_vector.top_words[:10],
            }
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Analysis failed: {str(e)}")
        print(error_details)
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'details': error_details if app.debug else None
        }), 500


@app.route('/transfer', methods=['POST'])
def transfer_style():
    """Transfer style to original text"""
    try:
        data = request.json
        original_text = data.get('original_text', '')
        style_name = data.get('style_name', 'custom_style')
        author_name = data.get('author_name', 'the target style')
        
        if not original_text:
            return jsonify({'error': 'No original text provided'}), 400
        
        if style_name not in style_cache:
            return jsonify({'error': 'Style not analyzed yet. Please analyze stylized text first.'}), 400
        
        print(f"\n[TRANSFER] Applying style: {style_name}")
        print(f"[TRANSFER] Original text: {original_text[:100]}...")
        
        # Get cached style vector
        style_vector = style_cache[style_name]
        
        # Perform style transfer
        styled_text = engine.transfer(original_text, style_vector, author_name)
        
        print(f"[TRANSFER] Transfer complete")
        
        return jsonify({
            'success': True,
            'original': original_text,
            'styled': styled_text,
            'style_name': style_name
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Transfer failed: {str(e)}")
        print(error_details)
        
        # Check if it's an Ollama connection error
        if 'Connection' in str(e) or 'ConnectionError' in str(e):
            return jsonify({
                'error': 'Cannot connect to Ollama. Please ensure Ollama is running on port 11434.',
                'details': str(e) if app.debug else None
            }), 503
        
        return jsonify({
            'error': f'Transfer failed: {str(e)}',
            'details': error_details if app.debug else None
        }), 500


@app.route('/get_styles', methods=['GET'])
def get_styles():
    """Get list of available styles"""
    return jsonify({
        'styles': list(style_cache.keys())
    })


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run on specified host and port (configurable via environment)
    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_RUN_PORT', 5002))
    debug_env = os.environ.get('FLASK_DEBUG', 'True')
    debug = True if str(debug_env).lower() in ('1', 'true', 'yes') else False

    app.run(host=host, port=port, debug=debug)
