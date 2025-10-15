
import AssistantAvatar from './components/AssistantAvatar';
import ChatInterface from './components/ChatInterface'; // <--- Se importa el componente del chat

// Objeto de colores para reutilizar en toda la app.
export const colors = {
  brand_dark: '#121212',
  brand_gray: '#1E1E1E',
  brand_ai_bubble: '#2C2C2E',
  brand_user_bubble: '#00A8A8',
  brand_accent: '#AF52DE',
  brand_light: '#EAEAEA',
  brand_muted: '#A0A0A0',
};

// Componente para inyectar los keyframes y estilos globales
const GlobalStyles = () => (
  <style>{`
    body {
      background-color: ${colors.brand_dark};
      color: ${colors.brand_light};
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      overflow: hidden;
    }

    @keyframes breathing-glow {
      0%, 100% { transform: scale(1); opacity: 0.8; }
      50% { transform: scale(1.05); opacity: 1; }
    }

    @keyframes listening-bars {
      0%, 100% { transform: scaleY(0.4); }
      50% { transform: scaleY(1); }
    }

    @keyframes thinking-dots {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1.0); }
    }

    /* Estilos para la barra de scroll */
    ::-webkit-scrollbar {
      width: 6px;
    }
    ::-webkit-scrollbar-track {
      background: ${colors.brand_gray};
    }
    ::-webkit-scrollbar-thumb {
      background: ${colors.brand_ai_bubble};
      border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #555;
    }
  `}</style>
);


function App() {
  return (
    <>
      <GlobalStyles />
      <div style={{
        display: 'flex',
        height: '100vh',
        width: '100%',
        backgroundColor: colors.brand_dark,
      }}>
        {/* Columna Izquierda: Avatar de la IA */}
        <div style={{
          width: '33.333333%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'flex-start',
          padding: '32px',
          borderRight: `1px solid ${colors.brand_gray}40`, // 40 es opacidad
        }}>
          <div style={{ width: '100%', marginBottom: '32px' }}>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 600, color: colors.brand_light }}>
              Mariantonieta
            </h1>
            <p style={{ fontSize: '0.875rem', color: colors.brand_muted }}>
              Your AI Voice Assistant
            </p>
          </div>
          
          <AssistantAvatar />

        </div>

        {/* Columna Derecha: Interfaz de Chat */}
        <div style={{ width: '66.666667%', display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Se reemplaza el placeholder por el componente ChatInterface */}
          <ChatInterface />
        </div>
      </div>
    </>
  );
}

export default App;

