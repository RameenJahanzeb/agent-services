# Contact Manager Agent Service

FastAPI backend that uses Gemini and LangGraph to interpret contact-management
requests. All contact data is accessed through the Next.js app's `/api/contacts`
routes; this service does not connect to the database.

## Setup

From this folder, create a virtual environment, install dependencies, and copy
`.env.example` to `.env`. Set `GEMINI_API_KEY` to a Gemini API key. The
`CONTACT_API_BASE_URL` defaults to `http://localhost:3000`; set
`CORS_ALLOWED_ORIGINS` to a comma-separated list when deploying.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Run the Next.js contact-manager app first, then start this service from the
`agent-service` folder:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Send `POST http://localhost:8000/chat` with JSON such as
`{"message":"Add Jane Doe, phone +12345678901"}`. The response contains
`reply`, `operation`, and `success`.
