# RAG Chatbot Frontend

Modern React + TypeScript + Tailwind CSS frontend for the RAG (Retrieval-Augmented Generation) Chatbot.

## Features

- Clean, minimal UI built with Tailwind CSS
- Real-time streaming chat responses
- Document upload support (PDF, DOCX, TXT, Markdown)
- Responsive design for all devices
- Type-safe with TypeScript

## Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Build

```bash
npm run build
```

The build output will be in the `dist/` directory.

### Environment Variables

Create a `.env.local` file:

```env
REACT_APP_API_URL=http://localhost:8000
```

## Project Structure

```
src/
├── components/       # React components
├── services/        # API service layer
├── types/           # TypeScript types
├── App.tsx          # Main app component
└── main.tsx         # Entry point
```

## Technologies

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## License

MIT
