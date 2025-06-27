import { useState } from 'react';
import './App.css';
import ImageCapture from './components/ImageCapture';

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading1, setLoading1] = useState(false);
  const [result1, setResult1] = useState(null);

  const handleImageUpload = (file) => {
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult1(null);
  };

  const handlePredict = async () => {
    if (!imageFile) return;
    const formData = new FormData();
    formData.append('file', imageFile);
    setLoading1(true);
    setResult1(null);
    try {
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult1(data);
    } catch (error) {
      setResult1({ error: 'Error connecting to backend.' });
    }
    setLoading1(false);
  };

  return (
    <div className="App" style={{ padding: 24 }}>
      <h1>Basket Manual Detector</h1>
      <ImageCapture onImageUpload={handleImageUpload} />
      {preview && (
        <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'flex-start', margin: '20px 0' }}>
          <div style={{ textAlign: 'center', marginRight: 32 }}>
            <img src={preview} alt="Preview" style={{ maxWidth: 300, border: '1px solid #ccc' }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 350 }}>
            <button onClick={handlePredict} disabled={loading1} style={{ marginBottom: 16 }}>
              {loading1 ? 'Analyzing...' : 'Check for Manual'}
            </button>
            {result1 && (
              <div style={{ marginTop: 12, width: 350 }}>
                <h2 style={{ textAlign: 'center' }}>Result:</h2>
                {result1.error ? (
                  <div style={{ background: '#ffdddd', color: 'red', padding: 12, borderRadius: 6, textAlign: 'center', fontWeight: 600 }}>
                    {result1.error}
                  </div>
                ) : (
                  <div style={{
                    background: result1.contains_manual ? '#d4edda' : '#f8d7da',
                    color: result1.contains_manual ? '#155724' : '#721c24',
                    padding: 16,
                    borderRadius: 8,
                    textAlign: 'center',
                    fontWeight: 600,
                    fontSize: '1.2em',
                    marginBottom: 8
                  }}>
                    {result1.contains_manual
                      ? 'Manual detected in freezer!'
                      : 'No manual found.'}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;