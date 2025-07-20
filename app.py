from flask import Flask, request, jsonify
import requests
import pandas as pd
import re
from flask_cors import CORS
import logging
from requests.exceptions import RequestException
import json
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://rangmanch.vercel.app", "http://localhost:3000"]}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set properly. Some features may not work.")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Get Apify API key
APIFY_API_KEY = os.getenv('APIFY_API_KEY')

@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment_api():
    """API endpoint to analyze Instagram post sentiment"""
    logger.debug("Received request: %s", request.get_json())
    
    data = request.get_json()
    
    if not data or 'post_url' not in data:
        logger.error("Missing post_url parameter")
        return jsonify({'error': 'Missing post_url parameter'}), 400
    
    post_url = data['post_url']
    
    # Check if Apify API key is available
    if not APIFY_API_KEY:
        logger.error("APIFY_API_KEY environment variable not set")
        return jsonify({
            'error': 'APIFY API key not configured. Please set the APIFY_API_KEY environment variable.',
            'debug_info': {
                'env_vars_available': list(os.environ.keys()),
                'apify_key_present': 'APIFY_API_KEY' in os.environ
            }
        }), 500
    
    try:
        post_info, comments = extract_comments_with_apify(post_url, APIFY_API_KEY)
        
        if isinstance(comments, str):
            logger.error("Apify error: %s", comments)
            return jsonify({'error': comments}), 400
        
        sentiment_results = analyze_sentiment_with_groq(comments)
        summary_data = generate_summary_data(post_info, sentiment_results)
        
        logger.debug("Returning response: %s", summary_data)
        return jsonify(summary_data)
    
    except Exception as e:
        logger.error("Server error: %s", str(e))
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

def extract_comments_with_apify(post_url: str, api_key: str, max_comments: int = 200) -> tuple:
    """
    Extract Instagram comments using Apify API
    """
    logger.debug("Processing post URL: %s", post_url)
    
    shortcode_match = re.search(r'instagram\.com/(?:p|reel)/([^/?]+)', post_url)
    if not shortcode_match:
        return None, "Invalid Instagram URL"
    
    shortcode = shortcode_match.group(1)
    logger.debug("Extracted shortcode: %s", shortcode)
    
    api_url = "https://api.apify.com/v2/actor-tasks/deaniket1234~instagram-comments-scraper-task/run-sync-get-dataset-items"
    params = {"token": api_key}
    
    payload = {
        "directUrls": [post_url],
        "maxComments": max_comments,
        "maxPages": 10
    }
    
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(3):
        try:
            response = requests.post(api_url, json=payload, headers=headers, params=params, timeout=30)
            if response.status_code in [200, 201]:
                break
            logger.warning("Attempt %d failed with status %d: %s", attempt + 1, response.status_code, response.text)
        except RequestException as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
            if attempt == 2:
                return None, f"API request failed after 3 attempts: {str(e)}"
    
    if response.status_code not in [200, 201]:
        return None, f"API request failed with status code {response.status_code}: {response.text}"
    
    data = response.json()
    logger.debug("Total items in response: %d", len(data))
    
    if not data:
        return None, "No data returned from API"
    
    post_data = None
    for item in data:
        if 'postData' in item or 'postInfo' in item:
            post_data = item.get('postData', item.get('postInfo', {}))
            break
        
        if (item.get('isPostAuthor', False) and 
            (item.get('isVerified', False) or item.get('isCaption', False))):
            post_data = item
            break
    
    if not post_data and len(data) > 0:
        post_data = data[0]
    
    post_info = {'username': 'Unknown', 'shortcode': shortcode}
    
    if post_data and isinstance(post_data, dict):
        if 'ownerUsername' in post_data:
            post_info['username'] = post_data.get('ownerUsername')
        elif 'owner' in post_data and 'username' in post_data.get('owner', {}):
            post_info['username'] = post_data.get('owner', {}).get('username')
    
    if post_info['username'] == 'Unknown':
        for item in data:
            if item.get('isPostAuthor', False) or item.get('isCaption', False):
                if 'ownerUsername' in item:
                    post_info['username'] = item.get('ownerUsername')
                    break
        
        if post_info['username'] == 'Unknown' and len(data) > 0 and 'ownerUsername' in data[0]:
            post_info['username'] = data[0].get('ownerUsername')
    
    comments = []
    for item in data:
        if item.get('isCaption', False):
            logger.debug("Skipping caption: %s", item)
            continue
        
        if 'text' in item or 'commentText' in item:
            comment_text = item.get('text', item.get('commentText', ''))
            comment_username = item.get('ownerUsername', item.get('username', 'Unknown'))
            comments.append({
                'username': comment_username,
                'text': comment_text
            })
            logger.debug("Collected comment %d: %s", len(comments), comment_text)
    
    logger.debug("Total comments extracted: %d", len(comments))
    return post_info, comments

def analyze_sentiment_with_groq(comments: list) -> list:
    """
    Analyze sentiment of comments using Groq AI API
    Processes comments in batches for efficiency
    """
    if not comments or len(comments) == 0:
        return []
    
    results = []
    batch_size = 10  # Process comments in batches to avoid token limits
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_results = process_batch_with_groq(batch)
        results.extend(batch_results)
    
    return results

def process_batch_with_groq(batch_comments: list) -> list:
    """
    Process a batch of comments with Groq AI API
    """
    if not batch_comments:
        return []
    
    if not groq_client:
        logger.error("Groq client not initialized - API key missing")
        # Return neutral sentiments as fallback
        return [
            {
                'username': comment.get('username', 'Unknown'),
                'text': comment.get('text', ''),
                'compound': 0.0,
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'reasoning': 'Groq API key not configured',
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
            for comment in batch_comments
        ]
    
    # Prepare the batch for analysis
    comment_texts = []
    for idx, comment in enumerate(batch_comments):
        text = comment.get('text', '').strip()
        if text:
            comment_texts.append(f"{idx}: {text}")
    
    if not comment_texts:
        return []
    
    # Create the prompt for batch processing
    prompt = f"""
Analyze the sentiment of the following comments. Each comment is prefixed with its index number.

Comments to analyze:
{chr(10).join(comment_texts)}

For each comment, provide the sentiment analysis in the following JSON format:
{{
    "index": <comment_index>,
    "sentiment": "<Positive|Negative|Neutral>",
    "compound": <float between -1 and 1>,
    "confidence": <float between 0 and 1>,
    "reasoning": "<brief explanation>"
}}

Rules:
1. Handle Hindi text, English text, emojis, and mixed languages appropriately
2. Consider cultural context and nuances
3. compound score: -1 (most negative) to +1 (most positive)
4. confidence: how confident you are in the analysis (0-1)
5. Provide response as a JSON array with one object per comment
6. If a comment is unclear or has no meaningful content, mark as "Neutral" with low confidence

Respond with only the JSON array, no additional text.
"""
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",  # or "mixtral-8x7b-32768" for better performance
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        logger.debug(f"Groq response: {response_text}")
        
        # Parse the JSON response
        try:
            sentiment_data = json.loads(response_text)
            if not isinstance(sentiment_data, list):
                sentiment_data = [sentiment_data]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response as JSON: {e}")
            # Fallback to neutral sentiment for the batch
            sentiment_data = [
                {
                    "index": idx,
                    "sentiment": "Neutral",
                    "compound": 0.0,
                    "confidence": 0.5,
                    "reasoning": "Failed to analyze"
                }
                for idx in range(len(batch_comments))
            ]
        
        # Map results back to comments
        results = []
        for comment_idx, comment in enumerate(batch_comments):
            # Find corresponding sentiment analysis
            sentiment_result = None
            for result in sentiment_data:
                if result.get('index') == comment_idx:
                    sentiment_result = result
                    break
            
            if not sentiment_result:
                # Fallback if no result found
                sentiment_result = {
                    "sentiment": "Neutral",
                    "compound": 0.0,
                    "confidence": 0.5,
                    "reasoning": "No analysis available"
                }
            
            analyzed_comment = {
                'username': comment.get('username', 'Unknown'),
                'text': comment.get('text', ''),
                'compound': float(sentiment_result.get('compound', 0.0)),
                'sentiment': sentiment_result.get('sentiment', 'Neutral'),
                'confidence': float(sentiment_result.get('confidence', 0.5)),
                'reasoning': sentiment_result.get('reasoning', ''),
                # Convert compound score to positive/negative/neutral scores for compatibility
                'positive': max(0, float(sentiment_result.get('compound', 0.0))),
                'negative': max(0, -float(sentiment_result.get('compound', 0.0))),
                'neutral': 1 - abs(float(sentiment_result.get('compound', 0.0)))
            }
            
            results.append(analyzed_comment)
        
        return results
        
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        # Return neutral sentiments as fallback
        return [
            {
                'username': comment.get('username', 'Unknown'),
                'text': comment.get('text', ''),
                'compound': 0.0,
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'reasoning': f'API Error: {str(e)}',
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
            for comment in batch_comments
        ]

def generate_summary_data(post_info: dict, sentiment_results: list) -> dict:
    """
    Generate a detailed summary of sentiment analysis results
    """
    if not sentiment_results or len(sentiment_results) == 0:
        return {"error": "No comments to analyze or no valid sentiment results."}
    
    df = pd.DataFrame(sentiment_results)
    total_comments = len(df)
    
    sentiment_counts = df['sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    
    positive_pct = (positive_count / total_comments) * 100 if total_comments > 0 else 0
    negative_pct = (negative_count / total_comments) * 100 if total_comments > 0 else 0
    neutral_pct = (neutral_count / total_comments) * 100 if total_comments > 0 else 0
    
    avg_sentiment = df['compound'].mean() if total_comments > 0 else 0
    avg_confidence = df['confidence'].mean() if total_comments > 0 else 0
    
    sentiment_strength = "Neutral"
    if avg_sentiment > 0.15:
        sentiment_strength = "Strongly Positive"
    elif avg_sentiment > 0.05:
        sentiment_strength = "Positive"
    elif avg_sentiment < -0.15:
        sentiment_strength = "Strongly Negative"
    elif avg_sentiment < -0.05:
        sentiment_strength = "Negative"
    
    most_positive_idx = df['compound'].idxmax() if total_comments > 0 else None
    most_negative_idx = df['compound'].idxmin() if total_comments > 0 else None
    
    most_positive = df.loc[most_positive_idx].to_dict() if most_positive_idx is not None else None
    most_negative = df.loc[most_negative_idx].to_dict() if most_negative_idx is not None else None
    
    return {
        "post_info": {
            "username": post_info.get('username', 'Unknown'),
            "shortcode": post_info.get('shortcode', 'Unknown')
        },
        "stats": {
            "total_comments": total_comments,
            "avg_sentiment": float(avg_sentiment),
            "avg_confidence": float(avg_confidence),
            "sentiment_strength": sentiment_strength,
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "neutral_count": int(neutral_count),
            "positive_pct": float(positive_pct),
            "negative_pct": float(negative_pct),
            "neutral_pct": float(neutral_pct)
        },
        "highlights": {
            "most_positive": most_positive,
            "most_negative": most_negative
        },
        "comments": sentiment_results
    }

# Add a debug endpoint to check environment variables
@app.route('/api/debug/env', methods=['GET'])
def debug_env():
    """Debug endpoint to check environment variables (remove in production)"""
    return jsonify({
        'apify_key_present': 'APIFY_API_KEY' in os.environ,
        'groq_key_present': 'GROQ_API_KEY' in os.environ,
        'env_vars': [key for key in os.environ.keys() if 'API' in key.upper()],
        'apify_key_length': len(os.getenv('APIFY_API_KEY', '')) if os.getenv('APIFY_API_KEY') else 0,
        'groq_key_length': len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
