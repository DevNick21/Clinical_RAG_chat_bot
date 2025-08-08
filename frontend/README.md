# Clinical RAG Chat Frontend

A React-based web interface for the Clinical RAG System.

## Features

- Interactive chat interface for querying medical data
- Markdown support for formatted responses
- Chat history persistence
- Responsive design for desktop and mobile
- Integration with the Clinical RAG backend

## Project Structure

```
frontend/
  ├── public/              # Static files
  ├── src/
  │   ├── components/      # Reusable React components
  │   ├── context/         # React context providers
  │   ├── hooks/           # Custom React hooks
  │   ├── pages/           # Page components
  │   ├── services/        # API and storage services
  │   ├── App.js           # Main app component
  │   ├── index.js         # Application entry point
  │   ├── index.css        # Global styles
  │   └── theme.js         # Material-UI theme configuration
  ├── package.json         # Project dependencies
  └── README.md            # Frontend documentation
```

## Setup and Installation

### Prerequisites

- Node.js (v16.x or higher, v18.x recommended for all dependencies)
- npm or yarn
- Backend API running (see ../api/README.md)

### Installation Steps

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Create a `.env` file with backend configuration:

```bash
REACT_APP_API_URL=http://localhost:5000/api
```

3. Start the development server:

```bash
npm start
```

4. Build for production:

```bash
npm run build
```

## Integration with Backend

This frontend communicates with the Flask API server located in the `../api` directory, which interfaces with the RAG_chat_pipeline system.

The connection is configured in `package.json` with the `proxy` setting which redirects API requests to the Flask backend running on port 5000:

```json
"proxy": "http://localhost:5000"
```

## Technologies Used

- React 18
- Material-UI (MUI)
- React Router
- Axios
- React Markdown
