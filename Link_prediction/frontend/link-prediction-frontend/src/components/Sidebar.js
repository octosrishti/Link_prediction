import React, { useState } from "react";

const PredictionForm = () => {
  const [src, setSrc] = useState("");
  const [dst, setDst] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    setError("");
    setResult(null);
    try {
      const response = await fetch(
        `http://localhost:8000/predict_explain?src=${src}&dst=${dst}`
      );
      if (!response.ok) throw new Error("Something went wrong");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial", maxWidth: "700px", margin: "auto" }}>
      <h2>🔗 Link Prediction with Explanation</h2>
      
      <div style={{ marginBottom: "15px" }}>
        <input
          type="number"
          placeholder="Source Node ID"
          value={src}
          onChange={(e) => setSrc(e.target.value)}
          style={{ padding: "8px", marginRight: "10px", width: "150px" }}
        />
        <input
          type="number"
          placeholder="Target Node ID"
          value={dst}
          onChange={(e) => setDst(e.target.value)}
          style={{ padding: "8px", marginRight: "10px", width: "150px" }}
        />
        <button
          onClick={handlePredict}
          style={{
            padding: "8px 16px",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Predict
        </button>
      </div>

      {error && <p style={{ color: "red" }}> {error}</p>}

      {result && (
        <div style={{ marginTop: "20px", background: "#f4f4f4", padding: "20px", borderRadius: "8px" }}>
          <h3>Prediction Result</h3>
          <p><strong>Source:</strong> {result.source}</p>
          <p><strong>Target:</strong> {result.target}</p>
          <p><strong>Prediction Score:</strong> {result.prediction_score.toFixed(4)}</p>

          <h4>Explanation Metrics</h4>
          <ul>
            {Object.entries(result.explanation).map(([key, value]) => (
              <li key={key}><strong>{key}:</strong> {value}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
