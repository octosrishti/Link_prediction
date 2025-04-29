// import React, { useState } from 'react';

// const HyperparameterDashboard = () => {
//     const [hyperparameters, setHyperparameters] = useState(null);
//     const [loading, setLoading] = useState(false);

//     const fetchHyperparameters = async () => {
//         try {
//             setLoading(true);
//             const response = await fetch("http://localhost:8000/optuna_hyperopt", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//             });
//             const data = await response.json();
//             setHyperparameters(data.best_hyperparameters);
//         } catch (error) {
//             console.error("Error fetching hyperparameters:", error);
//         } finally {
//             setLoading(false);
//         }
//     };

//     return (
//         <div style={{ fontFamily: 'Arial', padding: '1rem' }}>
//             <h1>Hyperparameter Tuning Results</h1>
//             <button onClick={fetchHyperparameters} disabled={loading}>
//                 {loading ? "Fetching..." : "Get Hyperparameters"}
//             </button>

//             {hyperparameters ? (
//                 <div style={{ marginTop: '1rem' }}>
//                     <h2>Best Hyperparameters</h2>
//                     <ul>
//                         {Object.entries(hyperparameters).map(([key, value]) => (
//                             <li key={key}>
//                                 <strong>{key}:</strong> {value}
//                             </li>
//                         ))}
//                     </ul>
//                 </div>
//             ) : (
//                 <p style={{ marginTop: '1rem' }}>No hyperparameters fetched yet.</p>
//             )}
//         </div>
//     );
// };

// export default HyperparameterDashboard;
import React, { useState } from "react";

const HyperparameterDashboard = () => {
  const [hyperparameters, setHyperparameters] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchHyperparameters = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/optuna_hyperopt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();
      setHyperparameters(data.best_hyperparameters);
    } catch (error) {
      console.error("Error fetching hyperparameters:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.heading}>🔧 Hyperparameter Tuner</h1>
        <button
          onClick={fetchHyperparameters}
          disabled={loading}
          style={{
            ...styles.button,
            ...(loading ? styles.buttonDisabled : {}),
          }}
        >
          {loading ? "Fetching..." : "Get Best Parameters 🚀"}
        </button>

        <div style={{ marginTop: "2rem" }}>
          {hyperparameters ? (
            <div>
              <h2 style={styles.subHeading}>📊 Optimal Parameters</h2>
              <ul style={styles.paramList}>
                {Object.entries(hyperparameters).map(([key, value]) => (
                  <li key={key} style={styles.paramItem}>
                    <span style={styles.key}>{key}</span>
                    <span style={styles.value}>{value}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p style={styles.noData}>No hyperparameters fetched yet.</p>
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    fontFamily: "'Segoe UI', sans-serif",
    minHeight: "50vh",
    background: "linear-gradient(135deg, #1f1c2c, #928dab)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "2rem",
  },
  card: {
    background: "rgba(255, 255, 255, 0.05)",
    borderRadius: "16px",
    padding: "2.5rem",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
    backdropFilter: "blur(14px)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    color: "#fff",
    maxWidth: "600px",
    width: "100%",
    animation: "fadeIn 0.8s ease-in-out",
  },
  heading: {
    fontSize: "2rem",
    marginBottom: "1rem",
    textAlign: "center",
  },
  subHeading: {
    fontSize: "1.4rem",
    marginBottom: "1rem",
    textAlign: "left",
    color: "#d0d0ff",
  },
  button: {
    padding: "0.75rem 1.5rem",
    fontSize: "1rem",
    background: "#00c9a7",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    transition: "0.3s ease",
    width: "100%",
  },
  buttonDisabled: {
    background: "#666",
    cursor: "not-allowed",
  },
  noData: {
    textAlign: "center",
    color: "#ccc",
    fontStyle: "italic",
  },
  paramList: {
    listStyle: "none",
    padding: 0,
    marginTop: "1rem",
  },
  paramItem: {
    display: "flex",
    justifyContent: "space-between",
    background: "rgba(255, 255, 255, 0.05)",
    padding: "0.75rem 1rem",
    borderRadius: "10px",
    marginBottom: "0.5rem",
    border: "1px solid rgba(255, 255, 255, 0.08)",
  },
  key: {
    fontWeight: "600",
    color: "#ffffffcc",
  },
  value: {
    color: "#00c9a7",
  },
};

export default HyperparameterDashboard;
