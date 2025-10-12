import React, { useCallback, useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
const api = import.meta.env.VITE_API_URL as string;

export const FaceLogin: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [userId, setUserId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const captureAndLogin = useCallback(async () => {
    setError(null); setResult(null);
    if (!userId.trim()) return setError("Ingresa el user_id.");
    const cam = webcamRef.current;
    const dataUrl = cam?.getScreenshot();
    if (!dataUrl) return setError("No se pudo capturar la imagen.");
    const blob = await (await fetch(dataUrl)).blob();
    const file = new File([blob], "selfie.jpg", { type: "image/jpeg" });

    const form = new FormData();
    form.append("user_id", userId.trim());
    form.append("file", file);

    setLoading(true);
    try {
      const { data } = await axios.post(`${api}/auth/login_multipart`, form);
      setResult(data);
    } catch (e:any) {
      setError(e?.response?.data?.detail || e.message || "Error desconocido");
    } finally { setLoading(false); }
  }, [userId]);

  return (
    <div style={{display:"grid", gap:12}}>
      <label style={{display:"grid", gap:6}}>
        <span>User ID</span>
        <input value={userId} onChange={(e)=>setUserId(e.target.value)}
          placeholder="maria"
          style={{padding:"8px 12px", borderRadius:8, border:"1px solid #32406a"}} />
      </label>

      <div style={{ borderRadius: 12, overflow: "hidden", border: "1px solid #283458" }}>
        <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
          style={{ width: "100%", height: "auto" }} />
      </div>

      <button onClick={captureAndLogin} disabled={loading}
        style={{padding:"10px 14px", borderRadius:10, border:0, background:"#4b6bff", color:"#fff"}}>
        {loading ? "Procesando..." : "Iniciar sesión con rostro"}
      </button>

      {error && <div style={{ background:"#2a1435", color:"#ffb3c7", padding:10, borderRadius:8 }}>{error}</div>}
      {result && <pre style={{whiteSpace:"pre-wrap"}}>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
};
