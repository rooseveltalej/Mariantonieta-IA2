import React from 'react';
import { colors } from '../App';

const ChatInput: React.FC = () => {
  // Estilo para los iconos SVG (micrófono y enviar)
  const iconStyle: React.CSSProperties = {
    width: '24px',
    height: '24px',
    fill: colors.brand_muted,
    cursor: 'pointer',
    transition: 'fill 0.2s ease',
  };

  return (
    <div style={{ padding: '16px 24px', borderTop: `1px solid ${colors.brand_gray}` }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        backgroundColor: colors.brand_ai_bubble,
        borderRadius: '99px',
        padding: '8px 16px',
      }}>
        {/* Aquí iría el icono del micrófono */}
        <svg viewBox="0 0 24 24" style={iconStyle}>
            <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z"></path>
        </svg>

        <input
          type="text"
          placeholder="Hello Mariantonieta, how are you today?"
          style={{
            flexGrow: 1,
            backgroundColor: 'transparent',
            border: 'none',
            outline: 'none',
            color: colors.brand_light,
            fontSize: '1rem',
            marginLeft: '12px',
            marginRight: '12px',
          }}
        />
        
        {/* Botón de Enviar */}
        <button style={{
            backgroundColor: colors.brand_user_bubble,
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            border: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
        }}>
          <svg viewBox="0 0 24 24" style={{ ...iconStyle, fill: colors.brand_dark, width: '20px', height: '20px' }}>
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
          </svg>
        </button>
      </div>
       {/* Indicador "Thinking..." */}
       <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '12px', gap: '8px' }}>
          <span style={{ width: '8px', height: '8px', backgroundColor: colors.brand_user_bubble, borderRadius: '50%'}}></span>
          <span style={{ color: colors.brand_muted, fontSize: '0.875rem' }}>Thinking...</span>
       </div>
    </div>
  );
};

export default ChatInput;