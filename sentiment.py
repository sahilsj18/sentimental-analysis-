import gspread
from oauth2client.service_account import ServiceAccountCredentials
from nltk.sentiment import SentimentIntensityAnalyzer

# Set up Google Sheets API credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Access the Google Sheets spreadsheet
sheet = client.open('Feedback Spreadsheet').sheet1

# Retrieve feedback data
feedback_data = sheet.get_all_records()

# Initialize SentimentIntensityAnalyzer from NLTK
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and update spreadsheet
for feedback in feedback_data:
    text = feedback['Feedback Text']
    sentiment_scores = sid.polarity_scores(text)
    sentiment = 'Positive' if sentiment_scores['compound'] > 0 else 'Negative' if sentiment_scores['compound'] < 0 else 'Neutral'
    sheet.update_cell(feedback['ID'], 3, sentiment)  # Assuming column 3 is for sentiment analysis results

print("Sentiment analysis completed and updated in the spreadsheet.")
