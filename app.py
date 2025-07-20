from flask import Flask, request, jsonify
import requests
import pandas as pd
import re
from flask_cors import CORS
import logging
from requests.exceptions import RequestException
import os
from typing import Dict, List, Tuple, Any
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://rangmanch.vercel.app", "http://localhost:3000"]}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'your-groq-api-key-here')  # Set this in your environment

# Apify API configuration
APIFY_API_KEY = os.getenv('APIFY_API_KEY')

@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment_api():
    """API endpoint to analyze Instagram post sentiment using Groq AI"""
    logger.debug("Received request: %s", request.get_json())
    
    data = request.get_json()
    
    if not data or 'post_url' not in data:
        logger.error("Missing post_url parameter")
        return jsonify({'error': 'Missing post_url parameter'}), 400
    
    post_url = data['post_url']
    
    try:
        # Extract comments using Apify
        post_info, comments = extract_comments_with_apify(post_url, APIFY_API_KEY)
        
        if isinstance(comments, str):
            logger.error("Apify error: %s", comments)
            return jsonify({'error': comments}), 400
        
        # Analyze sentiment using Groq AI
        sentiment_results = analyze_sentiment_with_groq(comments)
        summary_data = generate_summary_data(post_info, sentiment_results)
        
        logger.debug("Returning response: %s", summary_data)
        return jsonify(summary_data)
    
    except Exception as e:
        logger.error("Server error: %s", str(e))
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

def analyze_sentiment_with_groq(comments: List[Dict]) -> List[Dict]:
    """
    Analyze sentiment of comments using Groq AI API
    This handles Hindi text, emojis, and mixed languages better than VADER
    """
    if not comments or len(comments) == 0:
        return []
    
    results = []
    
    # Process comments in batches to avoid token limits
    batch_size = 10
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i + batch_size]
        
        try:
            # Prepare the prompt for Grok
            comment_texts = []
            for j, comment in enumerate(batch_comments):
                text = comment.get('text', '').strip()
                if text:
                    comment_texts.append(f"{j+1}. {text}")
            
            if not comment_texts:
                continue
            
            prompt = f"""
Analyze the sentiment of these Instagram comments. The comments may be in English, Hindi, or contain emojis and mixed languages. For each comment, provide:
1. Sentiment: Positive, Negative, or Neutral
2. Confidence score: 0.0 to 1.0
3. Brief reasoning (optional)

Comments to analyze:
{chr(10).join(comment_texts)}

Respond in JSON format like this:
{{
  "results": [
    {{
      "comment_number": 1,
      "sentiment": "Positive",
      "confidence": 0.85,
      "reasoning": "Expresses happiness with emojis"
    }},
    ...
  ]
}}

Focus on the emotional tone, context, and cultural nuances. Consider:
- Hindi words and their emotional context
- Emojis and their meanings
- Sarcasm and irony
- Cultural references
"""

            # Call Groq API
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert sentiment analyst who understands multiple languages including Hindi and English, as well as emoji meanings and cultural contexts."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "model": "llama-3.1-70b-versatile",
                "stream": False,
                "temperature": 0.1
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                groq_response = response.json()
                content = groq_response.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                try:
                    # Parse JSON response from Groq
                    sentiment_data = json.loads(content)
                    groq_results = sentiment_data.get('results', [])
                    
                    # Map results back to comments
                    for result in groq_results:
                        comment_idx = result.get('comment_number', 1) - 1
                        if 0 <= comment_idx < len(batch_comments):
                            comment = batch_comments[comment_idx]
                            sentiment = result.get('sentiment', 'Neutral').lower().capitalize()
                            confidence = float(result.get('confidence', 0.5))
                            
                            # Convert confidence to compound score (-1 to 1)
                            if sentiment == 'Positive':
                                compound = confidence * 0.8  # Scale to 0.8 max
                            elif sentiment == 'Negative':
                                compound = -confidence * 0.8  # Scale to -0.8 min
                            else:
                                compound = 0.0
                            
                            analyzed_comment = {
                                'username': comment.get('username', 'Unknown'),
                                'text': comment.get('text', ''),
                                'sentiment': sentiment,
                                'compound': compound,
                                'confidence': confidence,
                                'positive': confidence if sentiment == 'Positive' else 0.0,
                                'negative': confidence if sentiment == 'Negative' else 0.0,
                                'neutral': confidence if sentiment == 'Neutral' else 0.0,
                                'reasoning': result.get('reasoning', '')
                            }
                            results.append(analyzed_comment)
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Groq response as JSON, falling back to simple analysis")
                    # Fallback to basic sentiment analysis
                    for comment in batch_comments:
                        results.append({
                            'username': comment.get('username', 'Unknown'),
                            'text': comment.get('text', ''),
                            'sentiment': 'Neutral',
                            'compound': 0.0,
                            'confidence': 0.5,
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 1.0,
                            'reasoning': 'API parsing error'
                        })
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                # Fallback for this batch
                for comment in batch_comments:
                    results.append({
                        'username': comment.get('username', 'Unknown'),
                        'text': comment.get('text', ''),
                        'sentiment': 'Neutral',
                        'compound': 0.0,
                        'confidence': 0.5,
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 1.0,
                        'reasoning': 'API error'
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing batch {i//batch_size + 1}: {str(e)}")
            # Fallback for this batch
            for comment in batch_comments:
                results.append({
                    'username': comment.get('username', 'Unknown'),
                    'text': comment.get('text', ''),
                    'sentiment': 'Neutral',
                    'compound': 0.0,
                    'confidence': 0.5,
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'reasoning': 'Processing error'
                })
    
    return results

def extract_comments_with_apify(post_url: str, api_key: str, max_comments: int = 200) -> Tuple[Dict, List[Dict]]:
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

def generate_summary_data(post_info: Dict, sentiment_results: List[Dict]) -> Dict:
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
    avg_confidence = df['confidence'].mean() if total_comments > 0 and 'confidence' in df.columns else 0
    
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
