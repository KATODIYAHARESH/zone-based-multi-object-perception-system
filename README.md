# Zone-Based Multi-Object Perception System

A real-time multi-object perception platform for zone monitoring, object tracking, face analysis, analytics, alerts, and privacy controls. The repository combines a Next.js dashboard with a Python perception service.

## Features

- Live camera and simulated video analysis
- Zone creation and geometric overlap detection
- Multi-object tracking and dwell-time measurement
- Object detection with YOLO and optional open-vocabulary models
- Face detection, emotion, age, and face enrollment
- Zone entry/exit alerts and configurable notification rules
- Analytics dashboard, heatmaps, exports, and retention controls
- PostgreSQL-backed event storage
- Privacy controls for masking and sensitive-data handling
- Docker support for the Python perception service

## Repository Structure

```text
src/                 Next.js dashboard, APIs, database, and browser perception logic
perception_system/   Python perception service and Streamlit interface
public/              Browser models and MediaPipe WebAssembly assets
```

## Web Application

### Requirements

- Node.js 20 or later
- npm
- PostgreSQL

### Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Configure the database connection:

   ```bash
   export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/app_db"
   ```

   On Windows PowerShell, use:

   ```powershell
   $env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/app_db"
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open <http://localhost:3000>.

### Production Build

```bash
npm run build
npm start
```

### Database Migrations

```bash
npx drizzle-kit generate
npx drizzle-kit migrate
```

## Python Perception Service

The Python service includes detection, tracking, alerts, analytics, privacy, zones, and evaluation tools.

### Setup

```bash
cd perception_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start the Streamlit interface:

```bash
streamlit run main.py
```

Additional optional dependencies are listed in `perception_system/requirements-optional.txt`.

Run Docker Compose from the Python service directory:

```bash
cd perception_system
docker compose up
```

## Validation

```bash
npm run typecheck
npm run lint
npm run build
```

The Python tests can be run from the `perception_system` directory:

```bash
python -m pytest
```

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string used by the Next.js application |

## GitHub Repository

https://github.com/KATODIYAHARESH/zone-based-multi-object-perception-system

## License

No license has been specified for this project.
