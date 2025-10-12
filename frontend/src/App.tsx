import { FaceLogin } from "./components/FaceLogin";

export default function App() {
  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center"}}>
      <div style={{width:420, padding:24}}>
        <h1>Face Login</h1>
        <FaceLogin />
      </div>
    </div>
  );
}
