import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './fonts.css';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Guided demo tour for screen recording: open the Studio with ?demo (see src/demo/demoTour.ts).
if (new URLSearchParams(location.search).has('demo')) void import('./demo/demoTour').then((m) => m.startDemoTour());
