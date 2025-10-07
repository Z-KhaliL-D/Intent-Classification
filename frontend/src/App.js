import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [intent, setIntent] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setIntent("");
    setConfidence(null);

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", { text });
      setIntent(res.data.intent);
      setConfidence(res.data.confidence);
    } catch (err) {
      console.error(err);
      setIntent("Error connecting to backend");
    }

    setLoading(false);
  };

  const getIntentStyle = () => {
    if (intent.toLowerCase() === "oos") return { color: "red", fontWeight: "bold" };
    return { color: "#333", fontWeight: "bold" };
  };

  return (
    <div className="wrapper">
      <div className="card">
        <h1 className="title">Intent Classifier</h1>
        <p className="subtitle">Type a query and let the model guess your intent</p>

        <form className="form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. I want to change my username"
            className="input"
          />
          <button type="submit" className="button" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {intent && (
          <div className="result">
            Predicted Intent:{" "}
            <span style={getIntentStyle()}>{intent}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
