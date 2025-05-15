# COPSight 3.0

COPSight is a comprehensive customer service conversation evaluation and analysis tool that helps assess and improve customer service interactions.

## Features

- **Conversation Evaluation**: Automated evaluation of customer service conversations
- **Consolidated Reporting**: Detailed analysis and reporting of evaluation results
- **Score Card Generation**: Individual scorecards for customer service associates
- **Real-time Progress Tracking**: Live progress monitoring during evaluations
- **Automated Notifications**: Slack integration for scorecard delivery

## Prerequisites

- Python 3.9 or higher
- Google Cloud Platform account with necessary APIs enabled
- Google Service Account credentials
- Slack workspace with bot token
- Firebase account (for deployment)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd copsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file with the following variables:
```
GOOGLE_SERVICE_ACCOUNT_JSON=path/to/service-account.json
SHEET_ID=your-google-sheet-id
OUTPUT_SHEET_ID=your-output-sheet-id
PROMPTS_SHEET_ID=your-prompts-sheet-id
SLACK_BOT_TOKEN=your-slack-bot-token
```

## Configuration

1. **Google Sheets Setup**:
   - Create three Google Sheets:
     - Raw Data Sheet (for conversation data)
     - Output Sheet (for evaluation results)
     - Prompts Sheet (for evaluation criteria)

2. **Service Account Setup**:
   - Create a Google Cloud service account
   - Download the service account JSON key
   - Enable necessary Google APIs (Sheets, Drive)

3. **Slack Integration**:
   - Create a Slack app
   - Add necessary bot scopes
   - Install the app to your workspace

## Running Locally

1. Start the Streamlit app:
```bash
streamlit run Copsight_V3.py
```

2. Access the application at `http://localhost:8501`

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t copsight-app .
```

2. Run the container:
```bash
docker run -p 8501:8501 copsight-app
```

## Firebase Deployment

1. Install Firebase CLI:
```bash
npm install -g firebase-tools
```

2. Login to Firebase:
```bash
firebase login
```

3. Initialize Firebase:
```bash
firebase init
```

4. Deploy:
```bash
firebase deploy
```

## Project Structure

```
copsight/
├── Copsight_V3.py          # Main application file
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── firebase.json          # Firebase configuration
├── .dockerignore         # Docker ignore file
└── README.md             # This file
```

## Security Considerations

- Never commit sensitive credentials to version control
- Use environment variables for API keys and tokens
- Regularly rotate service account keys
- Implement proper access controls in Google Sheets
- Monitor API usage and quotas

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Support

For support, please contact [Your Contact Information]

## Version History

- 3.0.1 (May 2025)
  - Added consolidated reporting
  - Improved scorecard generation
  - Enhanced Slack integration
  - Added real-time progress tracking

## Acknowledgments

- Google Cloud Platform
- Streamlit
- Firebase
- Slack API 