import { useState } from 'react';
import './App.css';
import ImageCapture from './components/ImageCapture';

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [result1, setResult1] = useState(null);
  const [result2, setResult2] = useState(null);

  const handleImageUpload = (file) => {
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult1(null);
    setResult2(null);
  };

  const handlePredict = async (model) => {
    if (!imageFile) return;
    const formData = new FormData();
    formData.append('file', imageFile);
    if (model === 1) {
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
    } else if (model === 2) {
      setLoading2(true);
      setResult2(null);
      try {
        const response = await fetch('http://localhost:8000/predict_model2/', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        setResult2(data);
      } catch (error) {
        setResult2({ error: 'Error connecting to backend.' });
      }
      setLoading2(false);
    }
  };

  return (
    <div className="App" style={{ padding: 24 }}>
      <h1>Basket Manual Detector</h1>
      <ImageCapture onImageUpload={handleImageUpload} />
      {preview && (
        <div style={{ margin: '20px 0' }}>
          <img src={preview} alt="Preview" style={{ maxWidth: 300, border: '1px solid #ccc' }} />
        </div>
      )}
      {imageFile && (
        <div style={{ display: 'flex', gap: '2rem', marginBottom: 24 }}>
          <div>
            <button onClick={() => handlePredict(1)} disabled={loading1}>
              {loading1 ? 'Analyzing...' : 'Model 1'}
            </button>
            {result1 && (
              <div style={{ marginTop: 12 }}>
                <h2>Model 1 Result:</h2>
                {result1.error ? (
                  <p style={{ color: 'red' }}>{result1.error}</p>
                ) : (
                  <>
                    <p>
                      {result1.contains_manual
                        ? 'Manual detected in freezer!'
                        : 'No manual found.'}
                    </p>
                    {'raw_prediction' in result1 && (
                      <p style={{ fontSize: '0.9em', color: '#888' }}>
                        Raw prediction value: {result1.raw_prediction}
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
          <div>
            <button onClick={() => handlePredict(2)} disabled={loading2}>
              {loading2 ? 'Analyzing...' : 'Model 2'}
            </button>
            {result2 && (
              <div style={{ marginTop: 12 }}>
                <h2>Model 2 Result:</h2>
                {result2.error ? (
                  <p style={{ color: 'red' }}>{result2.error}</p>
                ) : (
                  <>
                    <p>
                      {result2.contains_manual
                        ? 'Manual detected in freezer!'
                        : 'No manual found.'}
                    </p>
                    {'raw_prediction' in result2 && (
                      <p style={{ fontSize: '0.9em', color: '#888' }}>
                        Raw prediction value: {result2.raw_prediction}
                      </p>
                    )}
                  </>
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