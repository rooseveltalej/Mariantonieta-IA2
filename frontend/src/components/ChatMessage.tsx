import React, { type CSSProperties } from 'react';
import { colors } from '../App'; // Importamos los colores

// Definimos las propiedades que recibirá el componente
interface ChatMessageProps {
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ sender, text, timestamp }) => {
  const isUser = sender === 'user';

  // Estilo base para el contenedor del mensaje y la hora
  const wrapperStyle: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: isUser ? 'flex-end' : 'flex-start', // Alinea a la derecha para el usuario
    marginBottom: '16px',
    width: '100%',
  };

  // Estilo para la burbuja del mensaje
  const bubbleStyle: CSSProperties = {
    maxWidth: '70%',
    padding: '12px 16px',
    borderRadius: '18px',
    backgroundColor: isUser ? colors.brand_user_bubble : colors.brand_ai_bubble,
    color: colors.brand_light,
    // Bordes específicos para darle la forma de "burbuja"
    borderTopLeftRadius: isUser ? '18px' : '4px',
    borderTopRightRadius: isUser ? '4px' : '18px',
  };
  
  // Estilo para la marca de tiempo
  const timestampStyle: CSSProperties = {
    fontSize: '0.75rem',
    color: colors.brand_muted,
    marginTop: '6px',
    padding: '0 8px',
  };

  return (
    <div style={wrapperStyle}>
      <div style={bubbleStyle}>
        {text}
      </div>
      <span style={timestampStyle}>{timestamp}</span>
    </div>
  );
};

export default ChatMessage;