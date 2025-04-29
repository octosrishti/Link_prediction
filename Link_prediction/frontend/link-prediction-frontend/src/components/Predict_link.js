import React, { useState, useEffect, useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import CytoscapeComponent from 'react-cytoscapejs';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const TrainingDashboard = () => {
  const [epochData, setEpochData] = useState([]);
  const [gcnLossData, setGcnLossData] = useState([]);
  const [gatLossData, setGatLossData] = useState([]);
  const [node2vecLossData, setNode2vecLossData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('gcn');
  const [metricsHistory, setMetricsHistory] = useState({});
  const [ws, setWs] = useState(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [viewMode, setViewMode] = useState('none'); // 'none' | 'chart' | 'graph'
  const [trainingProgress, setTrainingProgress] = useState(0);

  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws/train_metrics');
    setWs(websocket);

    websocket.onopen = () => setSocketConnected(true);

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const { model, epoch, metrics } = data;

      // Update the loss data dynamically as training progresses
      if (model === 'gcn') {
        setGcnLossData((prev) => [...prev, metrics.loss || 0]);
      } else if (model === 'gat') {
        setGatLossData((prev) => [...prev, metrics.loss || 0]);
      } else if (model === 'node2vec_mlp') {
        setNode2vecLossData((prev) => [...prev, metrics.loss || 0]);
      }

      if (!epochData.includes(epoch)) {
        setEpochData((prev) => [...prev, epoch]);
      }

      setMetricsHistory((prev) => ({
        ...prev,
        [model]: [...(prev[model] || []), { epoch, loss: metrics.loss }],
      }));

      // Update the training progress in real-time
      setTrainingProgress((epoch / 50) * 100); // Assuming 50 epochs for simplicity
    };

    websocket.onclose = () => setSocketConnected(false);

    return () => websocket.close();
  }, [epochData]);

  const handleStartTraining = () => {
    if (ws) {
      ws.send(JSON.stringify({ type: 'select_model', model: selectedModel }));
    }
  };

  const prepareGraphNodes = () => {
    if (!metricsHistory[selectedModel]) return [];
    return metricsHistory[selectedModel].map((metric) => ({
      data: {
        id: `epoch-${metric.epoch}`,
        name: `Epoch ${metric.epoch}\nLoss: ${metric.loss.toFixed(4)}`,
      },
    }));
  };

  const prepareGraphEdges = () => {
    if (!metricsHistory[selectedModel]) return [];
    const metrics = metricsHistory[selectedModel];
    return metrics.slice(1).map((metric, idx) => ({
      data: {
        id: `edge-${idx}`,
        source: `epoch-${metrics[idx].epoch}`,
        target: `epoch-${metric.epoch}`,
      },
    }));
  };

  const elements = useMemo(
    () => [...prepareGraphNodes(), ...prepareGraphEdges()],
    [metricsHistory[selectedModel]]
  );

  const chartData = {
    labels: epochData,
    datasets: [
      {
        label: 'GCN Loss',
        data: gcnLossData,
        borderColor: '#36A2EB',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        fill: true,
      },
      {
        label: 'GAT Loss',
        data: gatLossData,
        borderColor: '#FF6384',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: true,
      },
      {
        label: 'Node2Vec Loss',
        data: node2vecLossData,
        borderColor: '#4BC0C0',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Epochs',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Loss',
        },
        min: 0,
      },
    },
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', padding: '1rem', background: 'linear-gradient(to bottom, #f5f7fa, #c3cfe2)' }}>
      <h1 style={{ textAlign: 'center', color: '#444' }}>
        {viewMode === 'chart' ? 'Loss Chart View' : viewMode === 'graph' ? 'Training Progress Graph View' : 'Model Training Dashboard'}
      </h1>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="model-select" style={{ marginRight: '0.5rem', fontWeight: 'bold' }}>Select Model:</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{ borderRadius: '5px', padding: '0.5rem', fontFamily: 'Arial' }}
        >
          <option value="gcn">GCN</option>
          <option value="gat">GAT</option>
          <option value="node2vec_mlp">node2vec_mlp</option>
        </select>
        <button
          onClick={handleStartTraining}
          style={{
            marginLeft: '1rem',
            padding: '0.5rem 1rem',
            borderRadius: '5px',
            backgroundColor: '#36A2EB',
            color: '#fff',
            cursor: 'pointer',
            fontWeight: 'bold',
            transition: 'all 0.3s ease',
          }}
          onMouseOver={(e) => (e.target.style.transform = 'scale(1.1)')}
          onMouseOut={(e) => (e.target.style.transform = 'scale(1)')}
        >
          Start Training
        </button>
      </div>

      <div style={{ fontWeight: 'bold', marginBottom: '1rem' }}>
        Connection Status: {socketConnected ? <span style={{ color: 'green' }}>✅ Connected</span> : <span style={{ color: 'red' }}>❌ Disconnected</span>}
      </div>

      <progress value={trainingProgress} max={100} style={{ width: '100%', height: '10px', borderRadius: '5px', marginBottom: '1rem' }} />

      <div>
        <button
          onClick={() => setViewMode('chart')}
          style={{
            marginRight: '1rem',
            padding: '0.5rem 1rem',
            borderRadius: '5px',
            backgroundColor: '#FF6384',
            color: '#fff',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
          }}
          onMouseOver={(e) => (e.target.style.transform = 'scale(1.1)')}
          onMouseOut={(e) => (e.target.style.transform = 'scale(1)')}
        >
          Show Chart View
        </button>
        <button
          onClick={() => setViewMode('graph')}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '5px',
            backgroundColor: '#36A2EB',
            color: '#fff',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
          }}
          onMouseOver={(e) => (e.target.style.transform = 'scale(1.1)')}
          onMouseOut={(e) => (e.target.style.transform = 'scale(1)')}
        >
          Show Graph View
        </button>
      </div>

      {viewMode === 'chart' && (
        <div style={{ marginTop: '2rem' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      {viewMode === 'graph' && (
        <div style={{ marginTop: '2rem' }}>
          <CytoscapeComponent
            elements={elements}
            style={{ width: '100%', height: '500px', backgroundColor: '#fff', borderRadius: '8px' }}
            layout={{ name: 'cose' }}
            stylesheet={[
              {
                selector: 'node',
                style: {
                  label: 'data(name)',
                  'background-color': '#36A2EB',
                  'font-size': 12,
                  color: '#fff',
                  width: 40,
                  height: 40,
                  'text-halign': 'center',
                  'text-valign': 'center',
                  'border-width': 2,
                  'border-color': '#FFA500',
                  'shadow-blur': 8,
                  'shadow-color': '#FFA500',
                },
              },
              {
                selector: 'edge',
                style: {
                  width: 2,
                  'line-color': '#888',
                  'curve-style': 'bezier',
                },
              },
            ]}
          />
        </div>
      )}
    </div>
  );
};

export default TrainingDashboard;
