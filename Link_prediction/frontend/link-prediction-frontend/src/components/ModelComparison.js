// export default ModelComparison;
import React, { useState } from "react";

// Component to fetch and display the model evaluation metrics
function ModelComparison() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMetrics = async (modelType) => {
    // Reset previous metrics and set loading to true
    setMetrics(null);  // Reset previous metrics
    setError(null);    // Reset previous error
    setLoading(true);  // Start loading state
    
    try {
      const response = await fetch("http://localhost:8000/evaluate_link_prediction/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_type: modelType }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch metrics");
      }

      const data = await response.json();
      console.log("Metrics received from backend:", data);
      setMetrics(data); // Set the new model's metrics
    } catch (error) {
      setError(error.message);  // Set error if something goes wrong
    } finally {
      setLoading(false);  // End loading state
    }
  };

  return (
    <div style={styles.container}>
      {/* <h1 style={styles.title}>Model Comparison</h1> */}
      <div style={styles.buttonGroup}>
        <button style={styles.button} onClick={() => fetchMetrics("gcn")}>Evaluate GCN</button>
        <button style={styles.button} onClick={() => fetchMetrics("gat")}>Evaluate GAT</button>
        <button
  style={styles.button}
  onClick={() => fetchMetrics("node2vec_mlp")}
>
  Evaluate Node2Vec MLP
</button>

      </div>

      {loading && <p style={styles.loading}>⏳ Loading...</p>}
      {error && <p style={styles.error}> {error}</p>}

      {metrics && (
  <div style={styles.metricsBox}>
    <h2 style={styles.subtitle}> Evaluation Metrics</h2>

    {/* Conditional rendering based on available metrics */}
    {metrics.AUC !== undefined && metrics.AP !== undefined ? (
      <>
        {/* AUC Metrics */}
        <div style={styles.progressBarContainer}>
          <div style={styles.progressBarLabel}>AUC</div>
          <div style={styles.progressBarWrapper}>
            <div
              style={{
                ...styles.progressBar,
                width: `${metrics.AUC * 100}%`,
                backgroundColor: "#4caf50",
              }}
            ></div>
          </div>
          <div style={styles.progressText}>{metrics.AUC.toFixed(4)}</div>
        </div>

        {/* AP Metrics */}
        <div style={styles.progressBarContainer}>
          <div style={styles.progressBarLabel}>Average Precision (AP)</div>
          <div style={styles.progressBarWrapper}>
            <div
              style={{
                ...styles.progressBar,
                width: `${metrics.AP * 100}%`,
                backgroundColor: "#2196f3",
              }}
            ></div>
          </div>
          <div style={styles.progressText}>{metrics.AP.toFixed(4)}</div>
        </div>
      </>
    ) : (
      // Display Precision Score (Node2VecMLP-specific)
      <div style={styles.progressBarContainer}>
        <div style={styles.progressBarLabel}>Prediction Score (Precision)</div>
        <div style={styles.progressBarWrapper}>
          <div
            style={{
              ...styles.progressBar,
              width: `${metrics.prediction_score * 100}%`,
              backgroundColor: "#4caf50",
            }}
          ></div>
        </div>
        <div style={styles.progressText}>{metrics.prediction_score.toFixed(4)}</div>
      </div>
    )}
  </div>
)}

    </div>
  );
}

const styles = {
  container: {
    fontFamily: "Arial, sans-serif",
    padding: "40px",
    maxWidth: "700px",
    margin: "0 auto",
    minHeight: "50vh",
    // backgroundColor: "#f7f9fc",
     background: "linear-gradient(135deg, rgb(31, 28, 44), rgb(146, 141, 171))",
     boxShadow: "rgba(0, 0, 0, 0.4) 0px 8px 32px",
    borderRadius: "12px",
    // boxShadow: "0 4px 15px rgba(0,0,0,0.1)",
  },
  title: {
    textAlign: "center",
    marginBottom: "25px",
    color: "#333",
    fontSize: "28px",
    fontWeight: "bold",
  },
  subtitle: {
    color: "#333",
    marginBottom: "12px",
    fontSize: "20px",
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "center",
    gap: "15px",
    marginBottom: "20px",
  },
  button: {
    padding: "12px 25px",
    backgroundColor: "#4caf50",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: "bold",
    transition: "all 0.3s ease",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
  },
  buttonHover: {
    backgroundColor: "#45a049",
    boxShadow: "0 3px 8px rgba(0,0,0,0.15)",
  },
  loading: {
    textAlign: "center",
    color: "#333",
  },
  error: {
    color: "red",
    textAlign: "center",
  },
  metricsBox: {
    marginTop: "20px",
    backgroundColor: "#ffffff",
    borderRadius: "10px",
    padding: "25px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  },
  progressBarContainer: {
    marginBottom: "15px",
  },
  progressBarLabel: {
    fontWeight: "bold",
    color: "#333",
  },
  progressBarWrapper: {
    width: "100%",
    height: "15px",
    backgroundColor: "#e0e0e0",
    borderRadius: "8px",
    overflow: "hidden",
    marginBottom: "5px",
  },
  progressBar: {
    height: "100%",
    borderRadius: "8px",
  },
  progressText: {
    textAlign: "center",
    fontWeight: "bold",
    color: "#333",
  },
  tableContainer: {
    marginTop: "20px",
    backgroundColor: "#ffffff",
    padding: "15px",
    borderRadius: "8px",
    boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  th: {
    padding: "12px",
    backgroundColor: "#f4f4f4",
    color: "#333",
    textAlign: "left",
    fontWeight: "bold",
  },
  td: {
    padding: "10px",
    borderBottom: "1px solid #ddd",
    textAlign: "left",
  },
};

export default ModelComparison;
