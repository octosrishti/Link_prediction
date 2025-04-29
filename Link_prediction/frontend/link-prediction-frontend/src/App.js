import React from 'react';
import GraphView from './components/GraphView';
import Metrics from './components/Metrics';
import PredictionForm from './components/Sidebar';
import './App.css';
import LinkPredictor from './components/Metrics';
import useGraphData from './hooks/useGraphData';
import TrainingDashboard from './components/Predict_link';
import HyperparameterDashboard from './components/GraphView';
import ModelComparison from './components/ModelComparison';
const App = () => {
    return (
    //     <div className="container">
    //       <TrainingDashboard />
    //       <ModelComparison />
    //       <HyperparameterDashboard />
    //       <LinkPredictor />
    //       <PredictionForm />
    // </div>
    
    <div>
        <div className="card">
            <h2 className='predicttag'>Prediction Result</h2>
            <LinkPredictor />
        </div>
    <div className="dashboard-wrapper">
        {/* Section 1: Training & Comparison */}
        <div className="section">
            <div className="row">
            <div className="card full-width">
            <h2>Hyperparameter Dashboard</h2>
            <HyperparameterDashboard />
            </div>
            {/* <div className="card">
                <h2>Training Progress</h2>
                <TrainingDashboard />
            </div> */}
            <div className="card">
                <h2>Model Comparison</h2>
                <ModelComparison />
            </div>
            </div>
        </div>

        {/* Section 2: Hyperparameters */}
        <div className="section">
        <div className="card">
                <h2 className='traing'>Training Progress</h2>
                <TrainingDashboard />
            </div>
            {/* <div className="card full-width">
            <h2>Hyperparameter Dashboard</h2>
            <HyperparameterDashboard />
            </div> */}
        </div>

        {/* Section 3: Prediction */}
        <div className="section">
            <div className="row">
            <div className="card">
                <h2>Predict Link</h2>
                <PredictionForm />
            </div>
            </div>
        </div>
        </div>
</div>

    );
};

export default App;
