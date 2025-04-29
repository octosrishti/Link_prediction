import React, { useEffect, useState } from "react";
import axios from "axios";
import CytoscapeComponent from "react-cytoscapejs";

const LinkPredictor = () => {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [model, setModel] = useState("GCN");
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    axios
      .get("http://localhost:8000/load_graph")
      .then((res) => {
        const { nodes, edges } = res.data;

        const formattedNodes = nodes.map((id) => ({
          data: { id: id.toString() },
        }));
        const formattedEdges = edges.map(([src, tgt]) => ({
          data: {
            id: `e-${src}-${tgt}`,
            source: src.toString(),
            target: tgt.toString(),
            color: "#06b6d4",
            probability: "",
          },
        }));

        setGraphData({ nodes: formattedNodes, edges: formattedEdges });
      })
      .catch((error) => console.error("Error loading graph data:", error));
  }, []);

  const handlePredict = async () => {
    if (!source || !target) return;

    try {
      const res = await axios.post("http://localhost:8000/predict", {
        source,
        target,
        model,
      });

      const probability = res.data["Link Probability"];
      if (!probability) return;

      const edgeColor =
        probability > 0.8
          ? "#10b981"
          : probability > 0.5
          ? "#f59e0b"
          : "#ef4444";

      const newEdge = {
        data: {
          id: `e-${source}-${target}`,
          source,
          target,
          probability: probability.toFixed(4),
          color: edgeColor,
        },
      };

      const edgeExists = graphData.edges.some(
        (e) => e.data.id === newEdge.data.id
      );
      const updatedEdges = edgeExists
        ? graphData.edges.map((e) =>
            e.data.id === newEdge.data.id ? newEdge : e
          )
        : [...graphData.edges, newEdge];

      const updatedNodes = [...graphData.nodes];
      const nodeIds = new Set(updatedNodes.map((n) => n.data.id));
      if (!nodeIds.has(source)) updatedNodes.push({ data: { id: source } });
      if (!nodeIds.has(target)) updatedNodes.push({ data: { id: target } });

      setGraphData({ nodes: updatedNodes, edges: updatedEdges });
      setPrediction(probability);
      setShowModal(true); // Show graph after prediction
    } catch (error) {
      console.error("Prediction error:", error);
    }
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const openModal = () => {
    setShowModal(true);
  };

  return (
    <div
      style={{
        fontFamily: "Inter, sans-serif",
        background: "linear-gradient(135deg, #1f1c2c, #928dab)",
        color: "#f1f5f9",
        padding: "2rem",
        borderRadius: "1rem",
        marginTop: "2rem",
        boxShadow: "0 0 30px rgba(7, 121, 142, 0.2)",
      }}
    >
      <h2
        style={{
          textAlign: "center",
          color: "#06b6d4",
          fontSize: "1.75rem",
          marginBottom: "1.5rem",
        }}
      >
        🔗 Link Predictor
      </h2>

      <div
        style={{
          display: "flex",
          gap: "1rem",
          flexWrap: "wrap",
          justifyContent: "center",
          alignItems: "center",
          marginBottom: "1.5rem",
        }}
      >
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{
            padding: "0.6rem 1rem",
            borderRadius: "0.5rem",
            background: "#334155",
            color: "#f1f5f9",
            border: "1px solid #475569",
            fontWeight: "500",
          }}
        >
          <option value="GCN">GCN</option>
          <option value="GAT">GAT</option>
          <option value="Node2Vec">Node2Vec</option>
        </select>

        <input
          placeholder="Source Node"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          style={{
            padding: "0.6rem 1rem",
            borderRadius: "0.5rem",
            background: "#334155",
            color: "#f1f5f9",
            border: "1px solid #475569",
          }}
        />
        <input
          placeholder="Target Node"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          style={{
            padding: "0.6rem 1rem",
            borderRadius: "0.5rem",
            background: "#334155",
            color: "#f1f5f9",
            border: "1px solid #475569",
          }}
        />
        <button
          onClick={handlePredict}
          style={{
            padding: "0.6rem 1.5rem",
            borderRadius: "0.5rem",
            background: "#06b6d4",
            color: "#ffffff",
            fontWeight: "bold",
            border: "none",
            cursor: "pointer",
            boxShadow: "0 0 8px #06b6d4",
          }}
        >
          Predict
        </button>

        <button
          onClick={openModal}
          style={{
            padding: "0.6rem 1.5rem",
            borderRadius: "0.5rem",
            background: "#0ea5e9",
            color: "#ffffff",
            fontWeight: "bold",
            border: "none",
            cursor: "pointer",
            boxShadow: "0 0 8px #0ea5e9",
          }}
        >
          Show Graph
        </button>
      </div>

      {showModal && (
        <div
          style={{
            background: "#334155",
            border: "1px solid #06b6d4",
            padding: "1.5rem",
            borderRadius: "1rem",
            boxShadow: "0 0 20px #06b6d4",
            marginTop: "1rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <h3 style={{ color: "#facc15" }}>
              Link Probability:{" "}
              <span style={{ color: "#67e8f9" }}>
                {prediction?.toFixed(4) || "N/A"}
              </span>
            </h3>
            <button
              onClick={closeModal}
              style={{
                background: "#ef4444",
                color: "#fff",
                border: "none",
                borderRadius: "0.5rem",
                padding: "0.5rem 1rem",
                cursor: "pointer",
                fontWeight: "bold",
              }}
            >
              Close Graph
            </button>
          </div>

          <div style={{ width: "100%", height: "500px", marginTop: "1rem" }}>
            <CytoscapeComponent
              elements={graphData.nodes.concat(graphData.edges)}
              style={{ width: "100%", height: "100%" }}
              layout={{ name: "cose", animate: true }}
              stylesheet={[
                {
                  selector: "node",
                  style: {
                    label: "data(id)",
                    backgroundColor: "#3b82f6",
                    color: "#f1f5f9",
                    fontSize: 10,
                    textValign: "center",
                    textHalign: "center",
                    "text-outline-color": "#1e293b",
                    "text-outline-width": 1,
                  },
                },
                {
                  selector: "edge",
                  style: {
                    width: 2,
                    lineColor: "data(color)",
                    targetArrowColor: "data(color)",
                    targetArrowShape: "triangle",
                    curveStyle: "bezier",
                    label: "data(probability)",
                    fontSize: 8,
                    color: "#e2e8f0",
                    "text-outline-color": "#1e293b",
                    "text-outline-width": 1,
                  },
                },
                {
                  selector: ":selected",
                  style: {
                    "background-color": "#06b6d4",
                    "line-color": "#06b6d4",
                  },
                },
              ]}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default LinkPredictor;
