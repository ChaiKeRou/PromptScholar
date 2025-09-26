## PromptScholar

PromptScholar is a **locally deployed, web-based prototype** that leverages Large Language Models (LLMs) to make the literature review process faster and smarter.  

It enables researchers to:  
- ⚡ Generate effective search queries  
- 📑 Retrieve and filter relevant papers  
- 🔄 Refine queries based on results  

By integrating LLMs, PromptScholar makes literature reviews **quicker, more accurate, and less tedious**, helping users focus on insights rather than manual searching.  

---

## 📂 Project Structure
```

PromptScholar/
├── backend/       # Django backend API and services
├── frontend/      # React frontend for user interface
├── database/      # PostgreSQL database dump and configuration
└── README.md      # Project documentation

````

---

## ⚙️ Technical Guide

This guide provides step-by-step instructions for installing and running PromptScholar locally.  
It covers prerequisites, LLM integration via **Ollama**, and setup for both backend and frontend.

---

### 1. Prerequisites

Please ensure the following software is installed:

- **Python 3.8+** → [Download](https://www.python.org/downloads/)  
- **PostgreSQL** → [Download](https://www.postgresql.org/download/)  
- **Node.js & npm** → [Download](https://nodejs.org/)  
- **Ollama** → [Download](https://ollama.com/download)  

---

### 2. Large Language Models (LLMs) via Ollama

PromptScholar relies on **Ollama** for LLM integration. Ensure it is properly installed and running.

#### 2.1 Pull Required Models
```bash
ollama pull deepseek-r1:14b
ollama pull nomic-embed-text
````

#### 2.2 Start Ollama Server

* **GUI Mode**: Launch the Ollama app (service runs at [http://localhost:11433](http://localhost:11433))
* **CLI Mode**:

  ```bash
  ollama serve
  ```

#### 2.3 Verify Model is Running

```bash
ollama run deepseek-r1:14b
```

#### 2.4 Useful Commands

```bash
ollama list   # List available models
ollama ps     # Show running models
```

---

### 3. Backend Setup

#### 3.1 Navigate to Backend Folder

```bash
cd backend
```

#### 3.2 Create Virtual Environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Database Configuration

#### 4.1 Import Database

1. Obtain the SQL dump file (e.g., `init.sql` or `backup.dump`) from the project admin.
2. Create a PostgreSQL database:

   ```sql
   CREATE DATABASE django_project_db;
   ```

   *(If you use a different name, update `DB_NAME` in `.env`.)*

📖 [PostgreSQL Restore Docs](https://www.postgresql.org/docs/current/backup-dump.html)

#### 4.2 Create `.env` File

Inside the **backend** folder, create a `.env` file:

```env
SECRET_KEY=<your_django_secret_key>
DEBUG=True
DB_NAME=django_project_db
DB_USER=<your_postgres_username>
DB_PASSWORD=<your_postgres_password>
DB_HOST=localhost
DB_PORT=5432
```

---

### 5. Running the Backend

#### 5.1 Apply Migrations

```bash
python manage.py migrate
```

#### 5.2 Start Server

```bash
python manage.py runserver
```

Backend runs at 👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 6. Frontend Setup

#### 6.1 Navigate to Frontend Folder

```bash
cd frontend
```

#### 6.2 Install Dependencies

```bash
npm install
```

#### 6.3 Start Frontend

```bash
npm run dev
```

Frontend runs at 👉 [http://localhost:5173](http://localhost:5173)

⚠️ Ensure both the **backend** and **Ollama** are running.

---

## 🚀 Quick Start (TL;DR)

```bash
# Clone repo
git clone <your-repo-url>
cd PromptScholar

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# In a new terminal: Frontend setup
cd frontend
npm install
npm run dev

# Ollama (separate terminal, if not already running)
ollama serve
```

Then open 👉 [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and open a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

```

---

✅ This version is tailored for your **combined repo** (frontend + backend).  
Would you like me to also add a **“System Architecture” diagram (ASCII or Mermaid)** to visually show how frontend, backend, PostgreSQL, and Ollama connect? That would make your README stand out a lot.
```
