import React, { useState, useEffect, type CSSProperties } from 'react';
import { colors } from '../App'; // Importamos los colores desde App.tsx

type AvatarState = 'Idle' | 'Listening' | 'Thinking';

const AssistantAvatar: React.FC = () => {
  const [state, setState] = useState<AvatarState>('Idle');

  useEffect(() => {
    const states: AvatarState[] = ['Idle', 'Thinking', 'Listening'];
    let currentIndex = 0;
    const intervalId = setInterval(() => {
      currentIndex = (currentIndex + 1) % states.length;
      setState(states[currentIndex]);
    }, 4000);
    return () => clearInterval(intervalId);
  }, []);

  const baseCircleStyle: CSSProperties = {
    position: 'absolute',
    borderRadius: '50%',
    borderStyle: 'solid',
    transition: 'all 0.5s ease',
  };

  const faceElementStyle: CSSProperties = {
    borderRadius: '50%',
    transition: 'background-color 0.5s ease',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px', width: '100%', height: '100%' }}>
      {/* Contenedor principal del avatar */}
      <div style={{ position: 'relative', width: '288px', height: '288px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>

        {/* Círculo exterior */}
        <div style={{
          ...baseCircleStyle,
          width: '100%',
          height: '100%',
          borderWidth: '1px',
          borderColor: 'rgba(0, 168, 168, 0.3)', // brand_user_bubble con opacidad
          animation: state === 'Idle' ? 'breathing-glow 4s ease-in-out infinite' : 'none',
        }}></div>
        
        {/* Segundo círculo */}
        <div style={{
          ...baseCircleStyle,
          width: '85%',
          height: '85%',
          borderWidth: '1px',
          borderColor: state === 'Thinking' ? 'rgba(175, 82, 222, 0.5)' : 'rgba(0, 168, 168, 0.5)',
          animation: state === 'Idle' ? 'breathing-glow 4s ease-in-out infinite 0.7s' : 'none',
        }}></div>

        {/* Círculo central */}
        <div style={{
          ...baseCircleStyle,
          width: '60%',
          height: '60%',
          borderWidth: '1px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderColor: state === 'Thinking' ? 'rgba(175, 82, 222, 0.4)' : 'rgba(0, 168, 168, 0.3)',
          backgroundColor: state === 'Thinking' ? 'rgba(175, 82, 222, 0.2)' : 'rgba(30, 30, 30, 0.2)',
        }}>
          
          {/* ----- Contenido Interno ----- */}

          {(state === 'Idle' || state === 'Listening') && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: '16px' }}>
                <span style={{ ...faceElementStyle, width: '16px', height: '16px', backgroundColor: colors.brand_light }}></span>
                <span style={{ ...faceElementStyle, width: '16px', height: '16px', backgroundColor: colors.brand_light }}></span>
              </div>
              <div style={{ width: '40px', height: '4px', borderRadius: '99px', backgroundColor: colors.brand_light, marginTop: '4px' }}></div>
            </div>
          )}
          
          {state === 'Listening' && (
            <div style={{ position: 'absolute', display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: '6px', height: '50%', width: '80%', opacity: 0.8 }}>
                {[0.4, 0.2, 0, 0.3, 0.1].map(delay => (
                  <span key={delay} style={{ width: '8px', height: '100%', backgroundColor: colors.brand_user_bubble, borderRadius: '99px', animation: 'listening-bars 1.2s infinite ease-in-out', animationDelay: `-${delay}s` }}></span>
                ))}
            </div>
          )}

          {state === 'Thinking' && (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: '16px' }}>
                  <span style={{ ...faceElementStyle, width: '16px', height: '16px', backgroundColor: colors.brand_accent }}></span>
                  <span style={{ ...faceElementStyle, width: '16px', height: '16px', backgroundColor: colors.brand_accent }}></span>
                </div>
                <div style={{ width: '40px', height: '4px', borderRadius: '99px', backgroundColor: colors.brand_accent, marginTop: '4px' }}></div>
              </div>
              {Array.from({ length: 8 }).map((_, i) => (
                <span key={i} style={{ position: 'absolute', width: '8px', height: '8px', backgroundColor: colors.brand_accent, borderRadius: '50%', animation: 'thinking-dots 1.4s infinite ease-in-out both', transform: `rotate(${i * 45}deg) translateY(-80px)`, animationDelay: `${i * 0.15}s` }}></span>
              ))}
            </>
          )}

        </div>
      </div>

      <p style={{ color: colors.brand_light, fontWeight: 500, fontSize: '1.125rem', letterSpacing: '0.05em', textTransform: 'capitalize' }}>
        {state}
      </p>

    </div>
  );
};

export default AssistantAvatar;