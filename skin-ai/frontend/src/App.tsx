import React, { useState } from "react";

type Prediction = {
  label: string;
  index: number;
  probability: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export default function App() {
  const [name, setName] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [bp, setBp] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [medications, setMeds] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ prediction: Prediction; top_k: Prediction[]; disclaimer: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");

  const onFile = (f: File | null) => {
    setImage(f);
    setResult(null);
    setError(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!image) {
      setError("Please select an image.");
      return;
    }
    if (!name || age === "" || !bp) {
      setError("Please fill in name, age, and blood pressure.");
      return;
    }
    const fd = new FormData();
    fd.append("image", image);
    fd.append("name", name);
    fd.append("age", String(age));
    fd.append("blood_pressure", bp);
    fd.append("symptoms", symptoms);
    fd.append("medications", medications);
    fd.append("api_key", apiKey);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status} ${txt}`);
      }
      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      setError(err.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", fontFamily: "system-ui, Arial" }}>
      <h1>Skin Condition Predictor</h1>
      <p style={{ color: "#b45309", background: "#fff7ed", padding: "8px 12px", borderRadius: 8 }}>
        Research/Education only. Not for diagnosis or treatment. Always consult a clinician.
      </p>

      <form onSubmit={submit} style={{ display: "grid", gap: 12, marginTop: 12 }}>
        <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
          <label>
            Patient name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Doe" required
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Age
            <input type="number" value={age} onChange={(e) => setAge(e.target.value ? Number(e.target.value) : "")}
              placeholder="42" required style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Blood pressure
            <input value={bp} onChange={(e) => setBp(e.target.value)} placeholder="120/80 mmHg" required
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Current medications
            <input value={medications} onChange={(e) => setMeds(e.target.value)} placeholder="(optional)"
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
        </div>
        <label>
          Symptoms / notes
          <textarea value={symptoms} onChange={(e) => setSymptoms(e.target.value)} placeholder="Itching, redness..."
            rows={3} style={{ width: "100%", padding: 8, marginTop: 4 }} />
        </label>
        <label>
          Lesion photo (jpg/png)
          <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            style={{ display: "block", marginTop: 4 }} />
        </label>

        {preview && (
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
            <img src={preview} alt="preview" style={{ maxWidth: 280, borderRadius: 8, border: "1px solid #ddd" }} />
            <div style={{ fontSize: 12, color: "#666" }}>
              Make sure the image is in focus and well-lit. Crop to the affected area if possible.
            </div>
          </div>
        )}

        <details>
          <summary>Advanced</summary>
          <label>
            API key (if configured on server)
            <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="optional"
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
        </details>

        <button type="submit" disabled={loading} style={{ padding: "10px 14px" }}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </form>

      {error && <p style={{ color: "#b91c1c" }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Results</h2>
          <p><b>Top prediction:</b> {result.prediction?.label} ({(result.prediction?.probability * 100).toFixed(1)}%)</p>
          <div style={{ maxWidth: 600 }}>
            {result.top_k.map((p: Prediction) => (
              <div key={p.index} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{p.label}</span>
                  <span>{(p.probability * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: 8, background: "#eee", borderRadius: 4 }}>
                  <div style={{ width: `${p.probability * 100}%`, height: 8, background: "#16a34a", borderRadius: 4 }} />
                </div>
              </div>
            ))}
          </div>
          <p style={{ color: "#555", marginTop: 12 }}>{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
